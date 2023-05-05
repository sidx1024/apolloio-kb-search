import { processData as getApolloKnowledgeData } from './apollo-knowledge/apollo-knowledge-processor.js';
import { readJSON, writeJSON, parseArguments } from './utils.js';

async function createApolloKnowledgeData(inputFile, outputFile) {
  console.log('Processing Apollo knowledge data...');
  const rawData = await readJSON(inputFile);
  const processedData = await getApolloKnowledgeData(rawData);
  writeJSON(outputFile, processedData);
}

async function main() {
  const args = parseArguments();

  const dataset = args['--dataset'];
  const inputFile = args['--input'];
  const outputFile = args['--output'];

  if (!dataset || !inputFile || !outputFile) {
    throw new Error(
      'Ensure all required arguments are passed: --dataset, --input, --output'
    );
  }

  if (dataset === 'apollo-knowledge') {
    await createApolloKnowledgeData(inputFile, outputFile);
  } else {
    throw new Error(`Unsupported dataset: ${dataset}`);
  }
}

main();
