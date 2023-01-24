import type { Tensor2D, Tensor, Rank } from '@tensorflow/tfjs-node';

export default class Normalizer {
  private readonly xMinMax: [Tensor<Rank>, Tensor<Rank>];
  private readonly yMinMax: [Tensor<Rank>, Tensor<Rank>];

  constructor(
    public readonly xTensor: Tensor2D,
    public readonly yTensor: Tensor2D,
  ) {
    this.xMinMax = [xTensor.min(), xTensor.max()];
    this.yMinMax = [yTensor.min(), yTensor.max()];
  }

  getTrainData(xTensor?: Tensor2D, yTensor?: Tensor2D): [Tensor2D, Tensor2D] {
    return [
      this.normalize(xTensor ?? this.xTensor, 'x'),
      this.normalize(yTensor ?? this.yTensor, 'y'),
    ];
  }

  normalizePredictData(xTensorPredict: Tensor2D): Tensor2D {
    return this.normalize(xTensorPredict, 'x');
  }

  unNormalizePrediction(predictionTensor: Tensor<Rank> | Tensor<Rank>[]): Tensor2D {
    const [min, max] = this.yMinMax;

    return (predictionTensor as Tensor<Rank>).mul(max.sub(min)).add(min);
  }

  predictionToTabularData(predictionTensor: Tensor<Rank> | Tensor<Rank>[]): number[] {
    const dataSynced = this.unNormalizePrediction(predictionTensor).dataSync();

    return [...dataSynced.valueOf()];
  }

  private normalize(tensor: Tensor2D, type: 'x' | 'y'): Tensor2D {
    const [min, max] = type === 'x' ? this.xMinMax : this.yMinMax;

    return tensor.sub(min).div(max.sub(min));
  }
}
