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

interface Journal {
  post_title: string;
  url: string;
  created_date: string;
  note: string;
}

const journals: Journal[] = JSON.parse(
  fs.readFileSync(path.join(__dirname, '../data/keshav-journals.json'), 'utf-8'),
);

async function vectorizeJournals() {
  try {
    const pineconeVectors = [];
    for (const journal of journals) {
      const input = `Title: ${journal.post_title}; Content: ${journal.note}`;
      console.log('Creating embedding for post', `${journal.post_title}`);

      const { data: embed } = await openai.createEmbedding({
        input,
        model: 'text-embedding-ada-002',
      });
      const vector = {
        input,
        metadata: {
          note: input,
          url: journal.url,
        },
      };
      const pineconeVector = {
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