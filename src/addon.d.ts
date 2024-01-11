declare module "llama-bindings" {
  export function systemInfo(): string;
  export function llamaLoadModelFromFile(
    pathModel: string,
    params: Buffer
  ): LlamaModel;
  export function llamaModelDesc(model: LlamaModel): string;
  export function llamaModelDefaultParams(): Buffer;
  export function llamaContextDefaultParams(): Buffer;
  export function llamaModelQuantizeDefaultParams(): Buffer;
  export function llamaBackendInit(numa: boolean): void;
  export function llamaBackendFree(): void;
  export function llamaFreeModel(model: LlamaModel): void;
  export function llamaNewContextWithModel(
    model: LlamaModel,
    params: Buffer
  ): LlamaContext;
  export function llamaFree(ctx: Buffer): void;
  export function llamaTokenize(
    ctx: LlamaContext, // Replace with the actual type
    text: string,
    add_bos: boolean
  ): number[];
  export function llamaDecode(ctx: LlamaContext, tokens: number[]): string;
}

type LlamaModel = Buffer;
type LlamaContext = Buffer;

enum LlamaModelKVOverrideType {
  LLAMA_KV_OVERRIDE_INT,
  LLAMA_KV_OVERRIDE_FLOAT,
  LLAMA_KV_OVERRIDE_BOOL,
}

interface LlamaModelKVOverride {
  key: string;
  tag: LlamaModelKVOverrideType;
  int_value?: number;
  float_value?: number;
  bool_value?: boolean;
}

type LlamaProgressCallback = (progress: number) => boolean;

interface LlamaModelParams {
  n_gpu_layers: number;
  main_gpu: number;
  tensor_split: number[]; // Assuming tensor_split is an array of floats
  progress_callback: LlamaProgressCallback;
  progress_callback_user_data: any; // Assuming this is a generic pointer, represented as any in TypeScript
  kv_overrides: LlamaModelKVOverride[];
  vocab_only: boolean;
  use_mmap: boolean;
  use_mlock: boolean;
}

enum LlamaRopeScalingType {
  LLAMA_ROPE_SCALING_UNSPECIFIED = -1,
  LLAMA_ROPE_SCALING_NONE = 0,
  LLAMA_ROPE_SCALING_LINEAR = 1,
  LLAMA_ROPE_SCALING_YARN = 2,
  LLAMA_ROPE_SCALING_MAX_VALUE = LLAMA_ROPE_SCALING_YARN,
}

enum GgmlType {}
// Define the enum values here

interface LlamaContextParams {
  seed: number;
  n_ctx: number;
  n_batch: number;
  n_threads: number;
  n_threads_batch: number;
  rope_scaling_type: LlamaRopeScalingType;
  rope_freq_base: number;
  rope_freq_scale: number;
  yarn_ext_factor: number;
  yarn_attn_factor: number;
  yarn_beta_fast: number;
  yarn_beta_slow: number;
  yarn_orig_ctx: number;
  type_k: GgmlType;
  type_v: GgmlType;
  mul_mat_q: boolean;
  logits_all: boolean;
  embedding: boolean;
  offload_kqv: boolean;
}
