import { Tensor2D, tensor2d } from '@tensorflow/tfjs-node';

interface AiManagerProps {
  xShape: number;
  yShape: number;
  saveModel: boolean;
  modelName: string;
}

export default class AiManager {
  constructor(public readonly props: AiManagerProps) {}

  makeTensor2D(data: number[], shape: 'x' | 'y'): Tensor2D {
    return tensor2d(data, [data.length, shape === 'x' ? this.props.xShape : this.props.yShape]);
  }
}
