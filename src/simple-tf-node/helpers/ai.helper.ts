import type { Tensor2D, Tensor, Rank } from '@tensorflow/tfjs-node';

export const getMinMax = (xTensor: Tensor2D, yTensor: Tensor2D): [[Tensor2D, Tensor2D], [Tensor2D, Tensor2D]] => (
  [
    [xTensor.min(), xTensor.max()],
    [yTensor.min(), yTensor.max()],
  ]
);

export const normalize = (tensor: Tensor2D, [min, max]: [Tensor2D, Tensor2D]): Tensor2D => {
  return tensor.sub(min).div(max.sub(min));
};

export const unNormalize = (tensor: Tensor<Rank> | Tensor<Rank>[], [min, max]: [Tensor2D, Tensor2D]): Tensor2D => {
  return (tensor as Tensor<Rank>).mul(max.sub(min)).add(min);
};

export const splitTestTrain = <T extends Array<unknown>>(array: T, ratio: number): [T, T] => {
  if (ratio > 1 || ratio < 0) {
    throw new Error('ratio must be between 0 and 1');
  }

  const trainLength = Math.floor(array.length * ratio);
  const train = array.slice(0, trainLength);
  const test = array.slice(trainLength);

  return [train, test] as [T, T];
};
