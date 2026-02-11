var philadelphia_area = ee.FeatureCollection('TIGER/2018/Counties')
  .filter(ee.Filter.eq('NAMELSAD', 'Philadelphia County'))
  .geometry();

var landsat8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(philadelphia_area)
  .filterDate('2021-01-01', '2021-12-31');

var addCloudPercentage = function(image) {
  var qa = image.select('QA_PIXEL');
  var cloudBitMask = (1 << 3);
  var cloudShadowBitMask = (1 << 4);
  var isCloudOrShadow = qa.bitwiseAnd(cloudBitMask).neq(0)
    .or(qa.bitwiseAnd(cloudShadowBitMask).neq(0));
  var cloudStats = isCloudOrShadow.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: philadelphia_area,
    scale: 30,
    maxPixels: 1e13
  });
  var cloudPct = ee.Number(cloudStats.get('QA_PIXEL')).multiply(100);
  return image.set('roi_cloud_pct', cloudPct);
};

var maskClouds = function(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
    .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask);
};

var processedCol = landsat8
  .map(addCloudPercentage)
  .filter(ee.Filter.lte('roi_cloud_pct', 5))
  .map(maskClouds);

var dates = processedCol.aggregate_array('system:time_start').map(function(time){
  return ee.Date(time).format('YYYY-MM-dd');
});

var dateFeatures = ee.FeatureCollection(dates.map(function(dateString){
  return ee.Feature(null, {'date': dateString});
}));

Export.table.toDrive({
  collection: dateFeatures,
  description: 'Philadelphia_LST_2021_Dates',
  fileFormat: 'CSV'
});

var lstCollectionCelsius = processedCol.map(function(image) {
  var lstKelvin = image.select('ST_B10').multiply(0.00341802).add(149.0);
  var lstCelsius = lstKelvin.subtract(273.15).rename('LST_Celsius');
  return lstCelsius.copyProperties(image, image.propertyNames());
});

var maxLST = lstCollectionCelsius.max().clip(philadelphia_area);

var lstVis = {
  min: 20, 
  max: 45, 
  palette: ['blue', 'green', 'yellow', 'orange', 'red']
};

Map.centerObject(philadelphia_area, 11);
Map.addLayer(maxLST, lstVis, '2021 Philadelphia Maximum LST');

var imageInfo = processedCol.map(function(image) {
  var path = image.get('WRS_PATH');
  var row = image.get('WRS_ROW');
  var date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd');
  var cloud = image.get('roi_cloud_pct');
  return image.set('Path', path).set('Row', row).set('Date', date).set('CloudPct', cloud);
});

Export.table.toDrive({
  collection: imageInfo,
  description: 'Philadelphia_Landsat8_ImageDetails_2021',
  fileFormat: 'CSV',
  selectors: ['system:index', 'Date', 'Path', 'Row', 'CloudPct']
});

Export.image.toDrive({
  image: maxLST,
  description: 'Philadelphia_Max_LST_2021',
  scale: 30,
  region: philadelphia_area,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
