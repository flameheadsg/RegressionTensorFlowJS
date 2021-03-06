require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../load-csv');
const LinearRegression = require('./linear-regression');

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg']
});

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1,
  iterations: 3,
  batchSize: 10
});

regression.train();
const r2 = regression.test(testFeatures, testLabels);
console.log("MSE History:", regression.mseHistory);
console.log("R^2:", r2);
console.log("\n");

regression.predict([
  [120, 2, 380]
]).print();
