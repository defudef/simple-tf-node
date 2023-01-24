/*
 * Predict horse power by providing acceleration and weight
 */
import * as tf from '@tensorflow/tfjs-node';
import { Normalizer, DataTransformer, DataLoader, AiManager } from 'simple-tf-node/tools';

interface Data {
  Name: string;
  Miles_per_Gallon: number;
  Cylinders: number;
  Displacement: number;
  Horsepower: number;
  Weight_in_lbs: number;
  Acceleration: number;
  Year: string;
  Origin: string;
}

const makeModel = (manager: AiManager): tf.Sequential => {
  const model = tf.sequential();
  
  // Hidden layers
  model.add(tf.layers.dense({ inputShape: [manager.props.xShape], units: 1, useBias: true }));

  // Output layer
  model.add(tf.layers.dense({ units: manager.props.yShape, useBias: true }));

  // Compile
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.adam(),
  });

  return model;
};

const run = async () => {
  const TRAIN_RATIO = 0.8;

  const manager = new AiManager({
    xShape: 2,
    yShape: 1,
    saveModel: true,
    modelName: 'example/cars',
  });

  const data = await DataLoader.fromJson<Data>('cars');

  const dataTransformer = new DataTransformer(data);

  dataTransformer
    .setCleanupFunction(car => (
      typeof car.Acceleration === 'number'
      && typeof car.Weight_in_lbs === 'number'
      && typeof car.Horsepower === 'number'
      && car.Acceleration > 0
      && car.Weight_in_lbs > 0
      && car.Horsepower > 0
    ))
    .setXFunction(car => [car.Acceleration, car.Weight_in_lbs])
    .setYFunction(car => [car.Horsepower]);

  const [[x_input, x_test], [y_input, y_test]] = dataTransformer.transform(TRAIN_RATIO);

  // create Normalizer (it also stores normalized train tensors)
  const normalizer = new Normalizer(
    manager.makeTensor2D(x_input, 'x'),
    manager.makeTensor2D(y_input, 'y'),
  );

  // Create and train model
  const model = makeModel(manager);

  await model.fit(...normalizer.getTrainData(), {
    epochs: 100,
    batchSize: 24,
  });

  // Test predictions
  const x_predict = normalizer.normalizePredictData(manager.makeTensor2D(x_test, 'x'));
  
  const prediction = model.predict(x_predict);

  // Print results
  const tableData = normalizer.predictionToTabularData(prediction).map((pred, index) => ({
    Acc: x_test[index][0],
    Weight: x_test[index][1],
    HP: y_test[index][0],
    'HP (Pred)': pred,
    Diff: Math.abs(y_test[index][0] - pred),
  }));

  const avg = tableData.reduce((acc, curr) => acc + curr.Diff, 0) / tableData.length;

  console.table(tableData);
  console.log('Average difference: ', avg);

  // Save model
  if (manager.props.saveModel) {
    await model.save(`file://./models/${manager.props.modelName}`);
  }
};

run();
