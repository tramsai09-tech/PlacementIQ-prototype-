import { GoogleGenerativeAI } from '@google/generative-ai';
import { logger } from './logger';

// Only real, verified production Gemini models — ordered fastest-to-slowest.
// Fake models (gemini-3.x-flash, etc.) have been removed to prevent guaranteed 404 retries.
export let DISCOVERED_MODELS: string[] = [
  'gemini-2.0-flash',
  'gemini-2.5-flash',
  'gemini-1.5-flash',
  'gemini-1.5-pro',
  'gemini-2.5-pro',
];

// Remembers the last model that responded successfully — used as a fast-path
// shortcut on the next call to avoid re-trying slower fallback models.
let lastWorkingModel: string | null = null;

interface KeyClient {
  client: GoogleGenerativeAI;
  rawKey: string;
  maskedKey: string;
}

let keyClients: KeyClient[] = [];
let currentKeyIndex = 0;
let diagnosticCompleted = false;

async function runModelDiagnostic(keyClient: KeyClient) {
  if (diagnosticCompleted) return;
  diagnosticCompleted = true;

  logger.info({ sdk: '@google/generative-ai', version: '0.21.0', key: keyClient.maskedKey }, '🔍 Running Gemini SDK Model Diagnostic...');

  try {
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${keyClient.rawKey}`);
    if (res.ok) {
      const data = await res.json() as { models?: Array<{ name: string; supportedGenerationMethods?: string[] }> };
      if (data.models && Array.isArray(data.models)) {
        const available = data.models
          .filter(m => m.supportedGenerationMethods?.includes('generateContent'))
          .map(m => m.name.replace(/^models\//, ''))
          // Keep only stable production text generation models, exclude experimental/tts/image variants
          .filter(m => !m.includes('-exp') && !m.includes('-preview') && !m.includes('-tts') && !m.includes('-image') && !m.includes('nano') && !m.includes('gemma') && !m.includes('lyria'));

        if (available.length > 0) {
          DISCOVERED_MODELS = Array.from(new Set([...available, 'gemini-1.5-flash', 'gemini-1.5-pro']));
          logger.info({ key: keyClient.maskedKey, totalModels: available.length, models: available }, '✅ Filtered Production Gemini Models for configured API key');
        }
      }
    } else {
      logger.warn({ status: res.status, statusText: res.statusText }, 'Could not list models via Google REST API endpoint');
    }
  } catch (err: any) {
    logger.warn({ err: err?.message || err }, 'Gemini model discovery fetch failed, using built-in stable model candidates');
  }

  // Execute quick test generation using the top model candidate
  try {
    const topModel = DISCOVERED_MODELS[0];
    const model = keyClient.client.getGenerativeModel({
      model: topModel,
      generationConfig: { responseMimeType: 'application/json' },
    });
    const result = await Promise.race([
      model.generateContent('Respond with {"status": "ok"}'),
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error('Test timeout')), 8000)),
    ]);
    logger.info({ model: topModel, response: result.response.text().trim() }, '🎉 Gemini Model Diagnostic Test Response Successful');
  } catch (e: any) {
    logger.warn({ err: e?.message || e }, 'Gemini test generation failed during diagnostic');
  }
}

function getKeyClients(): KeyClient[] {
  if (keyClients.length === 0) {
    const keysString = process.env.GEMINI_API_KEYS || process.env.GEMINI_API_KEY || '';
    const keys = keysString.split(',').map(k => k.trim()).filter(k => k.length > 0);

    if (keys.length === 0) {
      logger.warn('No GEMINI API keys found in environment variables. API calls will fail.');
      keys.push('INVALID_KEY');
    }

    keyClients = keys.map(key => ({
      client: new GoogleGenerativeAI(key),
      rawKey: key,
      maskedKey: key.length > 4 ? `****${key.slice(-4)}` : '****',
    }));

    logger.info(`Initialized Gemini AI with ${keyClients.length} load-balanced API key(s).`);
  }
  return keyClients;
}

export async function runGeminiDiagnostic() {
  if (process.env.NODE_ENV === 'development') {
    logger.info('Skipping Gemini startup diagnostic in development mode to optimize reload speed.');
    return;
  }
  const clients = getKeyClients();
  if (clients.length > 0 && clients[0].rawKey !== 'INVALID_KEY') {
    await runModelDiagnostic(clients[0]);
  }
}

export function getNextKeyClient(offset = 0): KeyClient {
  const clients = getKeyClients();
  const index = (currentKeyIndex + offset) % clients.length;
  currentKeyIndex = (currentKeyIndex + 1) % clients.length;
  return clients[index];
}

export async function generateJson<T>(
  prompt: string,
  maxRetries = 3,
  timeoutMs = 30000,
): Promise<T> {
  const startTime = Date.now();
  let lastError: any = null;

  const clients = getKeyClients();

  // Hard-cap at 3 attempts maximum to prevent extreme retry storms.
  // The old formula (DISCOVERED_MODELS.length * clients.length) could reach 14+ retries
  // each with up to 2500ms backoff = 35+ seconds of pure waiting before giving up.
  const maxAttempts = Math.min(maxRetries + 1, 3);

  // Build a prioritized model list: try lastWorkingModel first (fast-path),
  // then fall back through remaining known-good models.
  const candidateModels: string[] = [];
  if (lastWorkingModel && DISCOVERED_MODELS.includes(lastWorkingModel)) {
    candidateModels.push(lastWorkingModel);
  }
  for (const m of DISCOVERED_MODELS) {
    if (m !== lastWorkingModel) candidateModels.push(m);
  }

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const keyClient = getNextKeyClient(attempt);
    // Pick model from prioritized list — fast-path model first
    const modelName = candidateModels[attempt % candidateModels.length];

    const model = keyClient.client.getGenerativeModel({
      model: modelName,
      generationConfig: {
        responseMimeType: 'application/json',
        temperature: 0.1,
      },
    });

    try {
      const result = await Promise.race([
        model.generateContent(prompt),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error(`Gemini API timeout after ${timeoutMs}ms`)), timeoutMs),
        ),
      ]);

      const text = result.response.text();
      const durationMs = Date.now() - startTime;

      logger.info({
        model: modelName,
        apiKey: keyClient.maskedKey,
        durationMs,
        attempt: attempt + 1,
        status: 200,
      }, '⚡ Gemini AI Request Successful');

      // Cache the winning model for the next call (fast-path optimisation)
      lastWorkingModel = modelName;

      // Extract JSON from response (strip markdown fences if present)
      const jsonMatch = text.match(/```(?:json)?\n([\s\S]*?)\n```/) || text.match(/\[[\s\S]*\]|\{[\s\S]*\}/);
      const jsonString = jsonMatch ? (jsonMatch[1] || jsonMatch[0]) : text;

      return JSON.parse(jsonString.trim()) as T;
    } catch (error: any) {
      lastError = error;
      const durationMs = Date.now() - startTime;
      const errorMessage = error?.message || String(error);
      const is404 = errorMessage.includes('404') || errorMessage.includes('not found');

      if (is404) {
        // Permanently purge failing 404 model from both DISCOVERED_MODELS and candidateModels
        DISCOVERED_MODELS = DISCOVERED_MODELS.filter(m => m !== modelName);
        candidateModels.splice(candidateModels.indexOf(modelName), 1);
        if (DISCOVERED_MODELS.length === 0) {
          DISCOVERED_MODELS = ['gemini-1.5-flash', 'gemini-1.5-pro'];
        }
        // Clear cached model if it was the one that 404'd
        if (lastWorkingModel === modelName) lastWorkingModel = null;
      }

      // Check if error is non-retriable (e.g. invalid auth key, or bad request/blocked input)
      const isNonRetriable =
        (errorMessage.includes('400') && !errorMessage.includes('schema') && !errorMessage.includes('mimeType')) ||
        ((errorMessage.includes('401') || errorMessage.includes('403') || errorMessage.includes('API key not valid')) && clients.length <= 1);

      logger.warn({
        model: modelName,
        apiKey: keyClient.maskedKey,
        attempt: attempt + 1,
        maxAttempts,
        durationMs,
        is404,
        isNonRetriable,
        err: errorMessage,
      }, `Gemini AI Request Failed (${is404 ? 'Model 404 — Purged' : isNonRetriable ? 'Non-Retriable' : 'API Error'}). ${ attempt < maxAttempts - 1 ? 'Retrying next model...' : 'All attempts exhausted.'}`);

      if (isNonRetriable) {
        throw error;
      }

      // Shorter backoff: 500ms flat delay (no exponential growth up to 2500ms)
      if (attempt < maxAttempts - 1) {
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }
  }

  logger.error({ err: lastError?.message || lastError }, '❌ All Gemini AI candidate models exhausted after max retries.');
  throw lastError || new Error('Gemini AI request failed after all retries');
}
