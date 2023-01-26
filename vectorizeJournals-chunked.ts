/*
  to run this file use the following command:
  npx tsx scripts/vectorizeJournals.ts
*/

import 'cross-fetch/polyfill';
import fs from 'fs';
import path from 'path';
import config from '../config';
import getOpenAI from '../lib/openai';
import getPinecone from '../lib/pinecone';
import { PineconeGuideMetadata } from '../types';

const openai = getOpenAI();
const pinecone = getPinecone<PineconeGuideMetadata>({
  baseUrl: config.PINECONE_JOURNAL_BASE_URL,
  namespace: config.PINECONE_JOURNAL_NAMESPACE,
});

const journals: Journal[] = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../data/keshav-journals.json'), 'utf-8'),
);

interface Journal {
  post_title: string;
  url: string;
  created_date: string;
  note: string;
}

async function vectorizeJournals() {
  journals.map(async (journal) => {
    const text = journal.note;
    // split text into array of sentences
    const data = text.replace(/\r?\n|\r/g, "").split(/(?<=[.?!])\s+(?=[a-z])/i);
    const chunkedData = chunkData(data);

    try {
      const pineconeVectors = [];
      for (let chunk of chunkedData) {
        console.log(
          'Creating embedding for chunk',
          `${chunk.start}-${chunk.end}`,
        );
        const { data: embed } = await openai.createEmbedding({
          input: chunk.text,
          model: 'text-embedding-ada-002',
        });

        const vector = {
          input: chunk.text,
          metadata: {
            text: chunk.text,
            url: journal.url,
            title: journal.post_title,
            date: journal.created_date,
          },
        };

        const pineconeVector = {
          id: `${chunk.start}-${chunk.end}`,
          metadata: vector.metadata,
          values: embed.data[0].embedding,
        };
        pineconeVectors.push(pineconeVector);
      }
  
      await pinecone.upsert({
        vectors: pineconeVectors,
      });
  
      console.log('Done');
    } catch (error) {
      console.error(error);
    }
  });
}

function chunkData(data) {
  type Chunk = {
    start: number;
    end: number;
    text: string;
  };
  const chunkedData: Chunk[] = [];
  const windowSize = 4;
  const strideSize = 1;
  for (let i = 0; i < data.length; i += strideSize) {
    const startIndex = i;
    const endIndex = Math.min(i + windowSize, data.length);
    const text = data.slice(startIndex, endIndex).join(' ');

    const mergedData = {
      start: startIndex,
      end: endIndex,
      text,
    };
    chunkedData.push(mergedData);
  }
  return chunkedData;
}

vectorizeJournals().catch((error) => {
  console.error('Caught error:', error);
});