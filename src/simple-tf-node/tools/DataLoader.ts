import { node, Tensor3D } from '@tensorflow/tfjs-node';
import { readFileSync } from 'fs';

type Data = Record<string, any>;

export default class DataLoader {
  static async fromJson<T extends Data>(name: string): Promise<T[]> {
    const data = readFileSync(`${__dirname}/../../../data/${name}.json`);
    // const data = await import(`${__dirname}/../../../data/${name}.json`);

    return JSON.parse(data.toString()) satisfies T[];
  }

  static fromImage(name: string): Tensor3D {
    const image = readFileSync(`${__dirname}/../../../data/${name}`);

    return node.decodeImage(image, 3) as Tensor3D;
  }

  static async fromCsv<T extends Data>(name: string): Promise<T[]> {
    throw new Error('DataImporter.fromCsv: Method not implemented');
  }
}
