// Load uploaded river Shapefile vector file (manually import!)
var riverShapefile = table.geometry(); 

// Load MODIS dataset
var MOD09GA = ee.ImageCollection('MODIS/061/MOD09GA');
var MYD09GA = ee.ImageCollection('MODIS/061/MYD09GA');

// Defining time ranges
var startDate = '2005-05-01';
var endDate   = '2005-08-01';
var start = ee.Date(startDate);
var end = ee.Date(endDate);

var dates = ee.List.sequence(start.millis(), end.millis(), 86400000); //86,400,000 milliseconds equals one day.

// Define the NDSI formula
function calculateNDSI(image) {
  return image.normalizedDifference(['sur_refl_b04', 'sur_refl_b06']).rename('NDSI');
}

// Define river ice classification rules
function classifyRiverIce(image) {
  var scaledImage = image.multiply(0.0001);
  var ndsi = calculateNDSI(scaledImage);
  var band2 = scaledImage.select('sur_refl_b02');
  var band4 = scaledImage.select('sur_refl_b04');

  var riverIce = ndsi.gt(0.4)
    .and(band2.gt(0.1))
    .and(band4.gt(0.11))
    .rename('riverIce');
  return image.addBands(riverIce);
}

// Get Cloud Mask
function getCloudMask(image) {
  var state = image.select('state_1km');
  var cloudMask = state.bitwiseAnd(1 << 10).neq(0); // 0: no cloud; 1: cloud
  return cloudMask.rename('cloudMask');
}

function processImage(image) {
  var cloudMask = getCloudMask(image);
  var classifiedImage = classifyRiverIce(image);
  cloudMask = cloudMask.resample('bilinear').reproject({
    crs: image.select(0).projection(),
    scale: 500
  });
  return classifiedImage.addBands(cloudMask);
}

// Acquisition of synthetic day-by-day images
function synthesizeDailyImage(date) {
  var terraImage = MOD09GA.filterDate(date, ee.Date(date).advance(1, 'day'))
                          .filterBounds(riverShapefile)
                          .map(processImage)
                          .first();    
  var aquaImage = MYD09GA.filterDate(date, ee.Date(date).advance(1, 'day'))
                         .filterBounds(riverShapefile)
                         .map(processImage)
                         .first(); 
  var terraValidMask = MOD09GA.filterDate(date, ee.Date(date).advance(1, 'day'))
                              .filterBounds(riverShapefile)
                              .select('sur_refl_b02')
                              .first();
  
  var aquaValidMask = MYD09GA.filterDate(date, ee.Date(date).advance(1, 'day'))
                            .filterBounds(riverShapefile)
                            .select('sur_refl_b02')
                            .first();

  // // Combining the effective pixel masks of Terra and Aqua
  var validMask = terraValidMask.or(aquaValidMask);
  
  // Determine whether both Terra and Aqua have valid records
  var terraValid = terraValidMask.mask(); 
  var aquaValid = aquaValidMask.mask();   
  
  var bothValid = terraValid.and(aquaValid);
  
  var terraIce = terraImage.select('riverIce');
  var terraCloud = terraImage.select('cloudMask');
  
  var aquaIce = aquaImage.select('riverIce');
  var aquaCloud = aquaImage.select('cloudMask');
  
  var cloudOnly = ee.Image(0)
    .where(bothValid, terraCloud.eq(1).and(aquaCloud.eq(1))) 
    .where(terraValid.and(aquaValid.not()), terraCloud.eq(1)) 
    .where(aquaValid.and(terraValid.not()), aquaCloud.eq(1));
  

  var riverIce = ee.Image(0)
    .where(bothValid, terraIce.eq(1).or(aquaIce.eq(1))) 
    .where(terraValid.and(aquaValid.not()), terraIce.eq(1)) 
    .where(aquaValid.and(terraValid.not()), aquaIce.eq(1)); 

  var riverWaterMask = riverIce.not().and(cloudOnly.not());  

  var finalClassification = ee.Image(0).rename('classification').clip(riverShapefile);    

  finalClassification = finalClassification.where(riverWaterMask, 0); 
  finalClassification = finalClassification.where(riverIce, 255); 
  // finalClassification = finalClassification.where(riverIceWithCloud, 255); 
  finalClassification = finalClassification.where(cloudOnly, 10); 
  finalClassification = finalClassification.where(validMask.not(), ee.Image.constant(0).mask()); 
  
  return finalClassification.clip(riverShapefile);
}


  
// Cloud pixel processing based on 3-day continuity
function temporalContinuity3Days(imageList) {
  var imageListAsList = imageList.toList(imageList.size());
  var prevImage = ee.Image(imageListAsList.get(0));
  var currentImage = ee.Image(imageListAsList.get(1)); 
  var nextImage = ee.Image(imageListAsList.get(2)); 
  
  var prevClassification = prevImage.select('classification');
  var currentClassification = currentImage.select('classification');
  // var nextClassification = nextImage.select('classification');
  

  var prevRiverIce = prevClassification.eq(255); 
  // var nextRiverIce = nextClassification.eq(255);
  
  // // Rule (a): If both the previous day and the next day are identified as river ice, the cloudy pixel is classified as river ice.
  // var ruleA = prevRiverIce.and(nextRiverIce);
  
  // // Rule (b): If the previous day is identified as river ice and the next day is cloudy, the cloudy pixel is classified as river ice.
  // var ruleB = prevRiverIce.and(nextCloud);
  
  // Rule (c): In all other cases, retain the original cloudy pixel.
  // var ruleC = prevRiverIce.or(nextRiverIce);
  var ruleD = prevRiverIce;      // ruleD is as same as ruleA & ruleB
  
  var updatedClassification = currentClassification
    // .where(ruleA, 255) 
    // .where(ruleB, 255);  
    // .where(ruleC, 255); 
    .where(ruleD, 255);
  
  return currentImage.addBands(updatedClassification.rename('classification'), ['classification'], true);
}

function temporalContinuity5Days(imageList) {
  var imageListAsList = imageList.toList(imageList.size());
  var prevImage2 = ee.Image(imageListAsList.get(0));
  var prevImage1 = ee.Image(imageListAsList.get(1));
  var currentImage = ee.Image(imageListAsList.get(2));
  var nextImage1 = ee.Image(imageListAsList.get(3));
  var nextImage2 = ee.Image(imageListAsList.get(4));
  var day3CloudMask = currentImage.select('classification').eq(10); 
  
  var iceMaskList = ee.List([prevImage2, prevImage1, currentImage, nextImage1, nextImage2]).map(function(image) {
    return ee.Image(image).select('classification').eq(255); 
  });
  
  var iceSum = ee.ImageCollection(iceMaskList).sum();
  
  // If the 3rd day is cloud and more than 2 out of 5 consecutive days are identified as river ice, the cloud pixel on the 3rd day is set to river ice.
  var updatedDay3 = currentImage.select('classification')
    .where(day3CloudMask.and(iceSum.gte(2)), 255); 

  return currentImage.addBands(updatedDay3.rename('classification'), ['classification'], true);
}


// Day-by-day image processing and export
dates.getInfo().forEach(function(millis) {
  var date = ee.Date(millis);
  var dailyImage = synthesizeDailyImage(date);
  // print(dailyImage)
  
  // Construct 3-day and 5-day time windows.
  var imageList3Days = ee.ImageCollection([
    synthesizeDailyImage(date.advance(-1, 'day')), 
    dailyImage, 
    synthesizeDailyImage(date.advance(1, 'day'))  
  ]);
  var processedImage3Days = temporalContinuity3Days(imageList3Days);
  // print(processedImage3Days)
  
  var imageList5Days = ee.ImageCollection([
    synthesizeDailyImage(date.advance(-2, 'day')),
    synthesizeDailyImage(date.advance(-1, 'day')), 
    processedImage3Days, 
    synthesizeDailyImage(date.advance(1, 'day')),  
    synthesizeDailyImage(date.advance(2, 'day'))  
  ]);
  var processedImage5Days = temporalContinuity5Days(imageList5Days);
  var exported_image = processedImage5Days.select('classification');
  
  // Exporting images
  Export.image.toDrive({
    image: exported_image,
    description: 'MackenzieRiverIce_' + date.format('yyyyMMdd').getInfo(),
    folder: 'MackenzieRiverIce_MODIS_CloudRemoved',
    region: riverShapefile,
    scale: 500,
    crs: 'EPSG:3995',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
});