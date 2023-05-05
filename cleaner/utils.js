import fs from 'fs';
import path from 'path';

export function splitChunk(chunk, CHUNK_SIZE = 512) {
  let chunks = [];
  const parts = Math.ceil(chunk.body.length / CHUNK_SIZE);

  let currentIndex = 0;

  for (let i = 0; i < parts; i++) {
    let endIndex = currentIndex + CHUNK_SIZE;

    if (endIndex < chunk.body.length) {
      while (chunk.body[endIndex] !== ' ' && endIndex > currentIndex) {
        endIndex--;
      }
    } else {
      endIndex = chunk.body.length;
    }

    chunks.push({
      ...chunk,
      body: chunk.body.slice(currentIndex, endIndex),
    });

    currentIndex = endIndex + 1;
  }

  return chunks;
}

export function readJSON(name) {
  const absoluteName = path.resolve(name);
  console.log(`Reading ${absoluteName}...`);
  return JSON.parse(fs.readFileSync(absoluteName, 'utf-8'));
}

export function writeJSON(name, data) {
  const absoluteName = path.resolve(name);
  const json = JSON.stringify(data);
  console.log(
    `Writing ${absoluteName}, size: ${Math.round(
      json.length / 1024 / 1024
    )} MB...`
  );
  fs.writeFileSync(absoluteName, json, 'utf-8');
}

export function parseArguments() {
  const args = process.argv.slice(2);
  const parsedArgs = {};

  for (let i = 0; i < args.length; i += 2) {
    parsedArgs[args[i]] = args[i + 1];
  }

  return parsedArgs;
}
