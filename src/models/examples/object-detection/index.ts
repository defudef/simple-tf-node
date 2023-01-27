import * as mobilenet from '@tensorflow-models/mobilenet';
import { DataLoader } from 'simple-tf-node/tools';

const classify = async (images: string[]) => {
  console.log('Loading model...');
  const model = await mobilenet.load({
    version: 2,
    alpha: 1,
  });

  console.log('Model loaded. Classifying...');

  for await (const image of images) {
    const result = await model.classify(DataLoader.fromImage(image));

    console.log({ [image]: result });
  }
}

classify([
  'car.jpg',
  'cat.jpeg',
  'nature.jpg',
  'rat.jpg',
  'wedding.jpg'
].map(image => `images/${image}`));
