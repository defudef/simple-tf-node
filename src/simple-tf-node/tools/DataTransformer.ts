import { util } from '@tensorflow/tfjs-node';
import { splitTestTrain } from '../helpers/ai.helper';

export default class DataTransformer<
  IData extends Record<string, any>,
> {
  private cleanupFn: (element: IData) => boolean;
  private xFn: (element: IData) => number[];
  private yFn: (element: IData) => number[];

  constructor(
    private readonly data: IData[],
    private readonly shuffleEnabled: boolean = true,
  ) {}

  setCleanupFunction(fn: (element: IData) => boolean): this {
    this.cleanupFn = fn;

    return this;
  }

  setXFunction(fn: (element: IData) => number[]): this {
    this.xFn = fn;

    return this;
  }

  setYFunction(fn: (element: IData) => number[]): this {
    this.yFn = fn;

    return this;
  }

  /**
   * 
   * @param testRatio number between 0 and 1
   * @returns [[x_input, x_test], [y_input, y_test]]
   */
  transform(testRatio: number): [[number[], number[]], [number[], number[]]] {
    if (!this.cleanupFn || !this.xFn || !this.yFn) {
      throw new Error('You must set all functions before transforming data');
    }

    this.shuffleEnabled && util.shuffle(this.data);

    const filteredData = this.data.filter(element => this.cleanupFn(element));

    const xInputTest = splitTestTrain(filteredData.map(element => this.xFn(element)), testRatio);
    const yInputTest = splitTestTrain(filteredData.map(element => this.yFn(element)), testRatio);

    //@ts-ignore
    return [xInputTest, yInputTest];
  }
}
