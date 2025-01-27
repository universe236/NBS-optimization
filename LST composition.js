// Define study area
var philadelphia = ee.FeatureCollection('TIGER/2018/Counties')
 .filter(ee.Filter.eq('NAMELSAD', 'Philadelphia County'))
 .geometry();

// Define study boundary
var philadelphia_area = ee.FeatureCollection('TIGER/2018/Counties')
 .filter(ee.Filter.eq('NAMELSAD', 'Philadelphia County'))
 .geometry();

// Load Landsat 8 Collection 2 Level 2 dataset
var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
 .filterBounds(philadelphia_area)
 .filterDate('2021-01-01', '2021-12-31')
 .filter(ee.Filter.lt('CLOUD_COVER_LAND', 50));

var maskClouds = function(image) {
 var qa = image.select('QA_PIXEL');
 var cloudMask = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 5).neq(0));
 return image.updateMask(cloudMask.not());
};

var computeCloudPercentage = function(image) {
 var qa = image.select('QA_PIXEL');
 var cloud = qa.bitwiseAnd(1 << 3).neq(0).or(qa.bitwiseAnd(1 << 5).neq(0));
 var phillyCloud = cloud.clip(philadelphia_area);
 var cloudPixels = phillyCloud.reduceRegion({
   reducer: ee.Reducer.sum(),
   geometry: philadelphia_area,
   scale: 30,
   maxPixels: 1e13
 }).get('QA_PIXEL');
 var totalPixels = phillyCloud.reduceRegion({
   reducer: ee.Reducer.count(),
   geometry: philadelphia_area,
   scale: 30,
   maxPixels: 1e13
 }).get('QA_PIXEL');
 var cloudPercentage = ee.Algorithms.If(
   ee.Number(totalPixels).gt(0),
   ee.Number(cloudPixels).divide(ee.Number(totalPixels)).multiply(100),
   0
 );
 return image.set('cloud_percentage_philadelphia', cloudPercentage);
};

landsat8 = landsat8
 .map(maskClouds)
 .map(computeCloudPercentage)
 .filter(ee.Filter.lte('cloud_percentage_philadelphia', 5));

var dates = landsat8.aggregate_array('system:time_start').map(function(time){
 return ee.Date(time).format('YYYY-MM-dd');
});

print('Available image dates:', dates);

var dateFeatures = ee.FeatureCollection(
 dates.map(function(date){
   return ee.Feature(null, {'date': date});
 })
);

Export.table.toDrive({
 collection: dateFeatures,
 description: 'Philadelphia_LST_2021_Dates',
 fileFormat: 'CSV'
});

var lstCollection = landsat8.select('ST_B10');

var lstCollectionCelsius = lstCollection.map(function(image) {
 var lstKelvin = image.multiply(0.00341802).add(149.0);
 var lstCelsius = lstKelvin.subtract(273.15);
 return lstCelsius.copyProperties(image, image.propertyNames());
});

var maxLST = lstCollectionCelsius.max().clip(philadelphia_area);

var lstVis = {
 min: 20,
 max: 50,
 palette: ['blue', 'green', 'yellow', 'orange', 'red']
};

Map.centerObject(philadelphia_area, 12);
Map.addLayer(maxLST, lstVis, '2021 Philadelphia Maximum LST');

var imageInfo = landsat8.map(function(image) {
 var path = image.get('WRS_PATH');
 var row = image.get('WRS_ROW');
 var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
 return image.set('Path', path).set('Row', row).set('Date', date);
});

var imageDetails = imageInfo.map(function(image) {
 return ee.Feature(null, {
   'Path': image.get('Path'),
   'Row': image.get('Row'),
   'Date': image.get('Date')
 });
});

print('Image details:', imageDetails);

Export.table.toDrive({
 collection: ee.FeatureCollection(imageDetails),
 description: 'Philadelphia_Landsat8_ImageDetails_2021',
 fileFormat: 'CSV'
});

Export.image.toDrive({
 image: maxLST,
 description: 'Philadelphia_Max_LST_2021',
 scale: 30,
 region: philadelphia_area,
 crs: 'EPSG:4326',
 maxPixels: 1e13
});