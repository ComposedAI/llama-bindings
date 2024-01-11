#include <napi.h>
#include <common.h>
#include <llama.h>

class LLAMAContext : public Napi::ObjectWrap<LLAMAContext>
{
public:
    llama_model_params model_params;
    llama_model *model;

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
    }

    ~LLAMAContext()
    {
        llama_free_model(model);
    }

    static void init(Napi::Object exports)
    {
        exports.Set("LLAMAContext", DefineClass(exports.Env(), "LLAMAContext", {}));
    }
};

Napi::Value systemInfo(const Napi::CallbackInfo &info)
{
    return Napi::String::From(info.Env(), llama_print_system_info());
}

Napi::Value LlamaLoadModelFromFile(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 2)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    Napi::String pathModel = info[0].As<Napi::String>();
    llama_model_params *params = info[1].As<Napi::External<llama_model_params>>().Data();

    struct llama_model *result = llama_load_model_from_file(pathModel.Utf8Value().c_str(), *params);

    // Wrap the pointer in a Napi::External object and return it
    return Napi::External<llama_model>::New(env, result);
}

Napi::Value LlamaModelDesc(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1)
    {
        Napi::TypeError::New(env, "Wrong number of arguments")
            .ThrowAsJavaScriptException();
        return env.Null();
    }

    llama_model *model = info[0].As<Napi::External<llama_model>>().Data();
    char buf[1024]; // Adjust size as needed
    llama_model_desc(model, buf, sizeof(buf));

    return Napi::String::New(env, buf);
}

Napi::Value LlamaModelDefaultParams(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    struct llama_model_params params = llama_model_default_params();
    llama_model_params *result = new llama_model_params(params);
    return Napi::External<llama_model_params>::New(env, result);
}

Napi::Value LlamaContextDefaultParams(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    struct llama_context_params params = llama_context_default_params();
    llama_context_params *result = new llama_context_params(params);
    return Napi::External<llama_context_params>::New(env, result);
}

Napi::Value LlamaModelQuantizeDefaultParams(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    struct llama_model_quantize_params params = llama_model_quantize_default_params();
    llama_model_quantize_params *result = new llama_model_quantize_params(params);
    return Napi::External<llama_model_quantize_params>::New(env, result);
}

void LlamaBackendInit(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    bool numa = info[0].As<Napi::Boolean>();
    llama_backend_init(numa);
}

void LlamaBackendFree(const Napi::CallbackInfo &info)
{
    llama_backend_free();
}

void LlamaFreeModel(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    llama_model *model = info[0].As<Napi::External<llama_model>>().Data();
    llama_free_model(model);
}

Napi::Value LlamaNewContextWithModel(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    llama_model *model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context_params *params = info[1].As<Napi::External<llama_context_params>>().Data();
    struct llama_context *ctx = llama_new_context_with_model(model, *params);
    return Napi::External<llama_context>::New(env, ctx);
}

void LlamaFree(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();
    llama_context *ctx = info[0].As<Napi::External<llama_context>>().Data();
    llama_free(ctx);
}

Napi::Value LlamaTokenize(const Napi::CallbackInfo &info)
{
    Napi::Env env = info.Env();

    // Convert arguments from JavaScript to C++
    const struct llama_context *ctx = info[0].As<Napi::External<llama_context>>().Data();

    // Text to tokenize
    const char *text = info[1].As<Napi::String>().Utf8Value().c_str();

    // Other parameters
    bool add_bos = info[2].As<Napi::Boolean>();
    bool special = false;

    // Call the function
    std::vector<llama_token> tokens = llama_tokenize(ctx, text, add_bos);

    Napi::Uint32Array result = Napi::Uint32Array::New(info.Env(), tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        result[i] = static_cast<uint32_t>(tokens[i]);
    }

    return result;
}

Napi::Value LlamaDecode(const Napi::CallbackInfo &info)
{
    // Convert arguments from JavaScript to C++
    const struct llama_context *ctx = info[0].As<Napi::External<llama_context>>().Data();

    Napi::Uint32Array tokens = info[1].As<Napi::Uint32Array>();

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

Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    exports.Set(Napi::String::New(env, "systemInfo"),
                Napi::Function::New(env, systemInfo));
    LLAMAContext::init(exports);
    exports.Set("llamaLoadModelFromFile", Napi::Function::New(env, LlamaLoadModelFromFile));
    exports.Set("llamaModelDesc", Napi::Function::New(env, LlamaModelDesc));
    exports.Set("llamaModelDefaultParams", Napi::Function::New(env, LlamaModelDefaultParams));
    exports.Set("llamaContextDefaultParams", Napi::Function::New(env, LlamaContextDefaultParams));
    exports.Set("llamaModelQuantizeDefaultParams", Napi::Function::New(env, LlamaModelQuantizeDefaultParams));
    exports.Set("llamaBackendInit", Napi::Function::New(env, LlamaBackendInit));
    exports.Set("llamaBackendFree", Napi::Function::New(env, LlamaBackendFree));
    exports.Set("llamaFreeModel", Napi::Function::New(env, LlamaFreeModel));
    exports.Set("llamaNewContextWithModel", Napi::Function::New(env, LlamaNewContextWithModel));
    exports.Set("llamaFree", Napi::Function::New(env, LlamaFree));
    exports.Set("llamaTokenize", Napi::Function::New(env, LlamaTokenize));
    exports.Set("llamaDecode", Napi::Function::New(env, LlamaDecode));
    return exports;
}

NODE_API_MODULE(addon, Init)
