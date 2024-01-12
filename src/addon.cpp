#include <napi.h>
#include <common.h>
#include <llama.h>

class LLAMAContext : public Napi::ObjectWrap<LLAMAContext>
{
public:
    llama_model_params model_params;
    llama_model *model;
    llama_context_params context_params;
    llama_context *ctx;
    int n_cur = 0;

    LLAMAContext(const Napi::CallbackInfo &info) : Napi::ObjectWrap<LLAMAContext>(info)
    {
        model_params = llama_model_default_params();

        // Get the model path
        std::string modelPath = info[0].As<Napi::String>().Utf8Value();

        if (info.Length() > 1 && info[1].IsObject())
        {
            Napi::Object options = info[1].As<Napi::Object>();

            if (options.Has("gpuLayers"))
            {
                model_params.n_gpu_layers = options.Get("gpuLayers").As<Napi::Number>().Int32Value();
            }

            if (options.Has("vocabOnly"))
            {
                model_params.vocab_only = options.Get("vocabOnly").As<Napi::Boolean>().Value();
            }

            if (options.Has("useMmap"))
            {
                model_params.use_mmap = options.Get("useMmap").As<Napi::Boolean>().Value();
            }

            if (options.Has("useMlock"))
            {
                model_params.use_mlock = options.Get("useMlock").As<Napi::Boolean>().Value();
            }
        }

        llama_backend_init(false);
        model = llama_load_model_from_file(modelPath.c_str(), model_params);

        if (model == NULL)
        {
            Napi::Error::New(info.Env(), "Failed to load model").ThrowAsJavaScriptException();
            return;
        }

        context_params = llama_context_default_params();
        context_params.seed = -1;
        context_params.n_ctx = 4096;
        context_params.n_threads = 6;
        context_params.n_threads_batch == -1 ? context_params.n_threads : context_params.n_threads_batch;

        ctx = llama_new_context_with_model(model, context_params);
        Napi::MemoryManagement::AdjustExternalMemory(Env(), llama_get_state_size(ctx));
    }
    Napi::Value Encode(const Napi::CallbackInfo &info)
    {
        std::string text = info[0].As<Napi::String>().Utf8Value();

        std::vector<llama_token> tokens = llama_tokenize(ctx, text, false);

        Napi::Uint32Array result = Napi::Uint32Array::New(info.Env(), tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i)
        {
            result[i] = static_cast<uint32_t>(tokens[i]);
        }

        return result;
    }
    Napi::Value Decode(const Napi::CallbackInfo &info)
    {
        Napi::Uint32Array tokens = info[0].As<Napi::Uint32Array>();

        // Create a stringstream for accumulating the decoded string.
        std::stringstream ss;

        // Decode each token and accumulate the result.
        for (size_t i = 0; i < tokens.ElementLength(); i++)
        {
            const std::string piece = llama_token_to_piece(ctx, (llama_token)tokens[i]);

            if (piece.empty())
            {
                continue;
            }

            ss << piece;
        }

        return Napi::String::New(info.Env(), ss.str());
    }

    ~LLAMAContext()
    {
        llama_free_model(model);
    }

    static void init(Napi::Object exports)
    {
        std::initializer_list<Napi::ClassPropertyDescriptor<LLAMAContext>> static_props = {
            InstanceMethod("encode", &LLAMAContext::Encode),
            InstanceMethod("decode", &LLAMAContext::Decode),

        };
        exports.Set("LLAMAContext", DefineClass(exports.Env(), "LLAMAContext", static_props));
    }
};

Napi::Value systemInfo(const Napi::CallbackInfo &info)
{
    return Napi::String::From(info.Env(), llama_print_system_info());
}

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(Napi::String::New(env, "systemInfo"),
                Napi::Function::New(env, systemInfo));
    LLAMAContext::init(exports);
    return exports;
}

NODE_API_MODULE(addon, Init)
