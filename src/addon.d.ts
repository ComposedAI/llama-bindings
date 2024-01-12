declare module "llama-bindings" {
  export type LLAMAModelParams = {
    gpuLayers?: number;
    vocabOnly?: boolean;
    useMmap?: boolean;
    useMlock?: boolean;
  };

  export function systemInfo(): string;

  export class LLAMAContext {
    constructor(modelPath: string, options?: LLAMAModelParams);

    encode(text: string): Uint32Array;
    decode(tokens: Uint32Array): string;
  }
}
