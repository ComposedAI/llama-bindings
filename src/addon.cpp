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

    Napi::Value EncodeBatch(const Napi::CallbackInfo &info)
    {
        Napi::Array texts = info[0].As<Napi::Array>();

        std::vector<llama_token> tokens;
        for (size_t i = 0; i < texts.Length(); ++i)
        {
            Napi::Value val = texts.Get(i);
            if (!val.IsString())
            {
                Napi::TypeError::New(info.Env(), "Expected all elements of array to be strings").ThrowAsJavaScriptException();
                return info.Env().Undefined();
            }

            std::string text = val.As<Napi::String>().Utf8Value();
            std::vector<llama_token> textTokens = llama_tokenize(ctx, text, false);
            tokens.insert(tokens.end(), textTokens.begin(), textTokens.end());
        }

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

    Napi::Value DecodeBatch(const Napi::CallbackInfo &info)
    {
        Napi::Array tokens = info[0].As<Napi::Array>();

        // Create a stringstream for accumulating the decoded string.
        std::stringstream ss;

        // Decode each token and accumulate the result.
        for (size_t i = 0; i < tokens.Length(); i++)
        {
            Napi::Value val = tokens.Get(i);
            if (!val.IsNumber())
            {
                Napi::TypeError::New(info.Env(), "Expected all elements of array to be numbers").ThrowAsJavaScriptException();
                return info.Env().Undefined();
            }

            llama_token token = static_cast<llama_token>(val.As<Napi::Number>().Int32Value());
            const std::string piece = llama_token_to_piece(ctx, token);

            if (piece.empty())
            {
                continue;
            }

            ss << piece;
        }

        return Napi::String::New(info.Env(), ss.str());
    }

    Napi::Value Evaluate(const Napi::CallbackInfo &info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsString())
        {
            Napi::TypeError::New(env, "Expected a string as the first argument").ThrowAsJavaScriptException();
            return env.Null();
        }

        // Convert the argument from JavaScript to C++
        std::string prompt = info[0].As<Napi::String>().Utf8Value();

        // Create a stringstream for accumulating the decoded string.
        std::stringstream ss;

        // total length of the sequence including the prompt
        const int n_len = 32;

        // tokenize the prompt

        std::vector<llama_token> tokens_list;
        tokens_list = ::llama_tokenize(ctx, prompt, true);

        const int n_ctx = llama_n_ctx(ctx);
        const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

        LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

        // make sure the KV cache is big enough to hold all the prompt and generated tokens
        if (n_kv_req > n_ctx)
        {
            Napi::Error::New(env, "n_kv_req > n_ctx, the required KV cache size is not big enough").ThrowAsJavaScriptException();
            return env.Null();
        }

        // print the prompt token-by-token

        fprintf(stderr, "\n");

        for (auto id : tokens_list)
        {
            fprintf(stderr, "%s", llama_token_to_piece(ctx, id).c_str());
        }

        fflush(stderr);

        // create a llama_batch with size 512
        // we use this object to submit token data for decoding

        llama_batch batch = llama_batch_init(512, 0, 1);

        // evaluate the initial prompt
        for (size_t i = 0; i < tokens_list.size(); i++)
        {
            llama_batch_add(batch, tokens_list[i], i, {0}, false);
        }

        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0)
        {
            Napi::Error::New(env, "%s: llama_decode() failed").ThrowAsJavaScriptException();
            return env.Null();
        }

        // main loop

        int n_cur = batch.n_tokens;
        int n_decode = 0;

        const auto t_main_start = ggml_time_us();

        while (n_cur <= n_len)
        {
            // sample the next token
            {
                auto n_vocab = llama_n_vocab(model);
                auto *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

                std::vector<llama_token_data> candidates;
                candidates.reserve(n_vocab);

                for (llama_token token_id = 0; token_id < n_vocab; token_id++)
                {
                    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                }

                llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

                // sample the most likely token
                const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

                // is it an end of stream?
                if (new_token_id == llama_token_eos(model) || n_cur == n_len)
                {
                    LOG_TEE("\n");

                    break;
                }

                // Convert the token to a piece
                std::string piece = llama_token_to_piece(ctx, new_token_id);

                // Add the piece to the stringstream
                ss << piece;

                fflush(stdout);

                // prepare the next batch
                llama_batch_clear(batch);

                // push this new token for next evaluation
                llama_batch_add(batch, new_token_id, n_cur, {0}, true);

                n_decode += 1;
            }

            n_cur += 1;

            // evaluate the current batch with the transformer model
            if (llama_decode(ctx, batch))
            {
                Napi::Error::New(env, "%s: failed to eval, return code %d").ThrowAsJavaScriptException();
                return env.Null();
            }
        }

        LOG_TEE("\n");

        const auto t_main_end = ggml_time_us();

        LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
                __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

        llama_print_timings(ctx);

        // Convert the result from C++ to JavaScript
        return Napi::String::New(env, ss.str());
    }

    ~LLAMAContext()
    {
        llama_free(ctx);
        llama_free_model(model);
    }

    static void init(Napi::Object exports)
    {
        std::initializer_list<Napi::ClassPropertyDescriptor<LLAMAContext>> static_props = {
            InstanceMethod("encode", &LLAMAContext::Encode),
            InstanceMethod("encodeBatch", &LLAMAContext::EncodeBatch),
            InstanceMethod("decode", &LLAMAContext::Decode),
            InstanceMethod("decodeBatch", &LLAMAContext::DecodeBatch),
            InstanceMethod("evaluate", &LLAMAContext::Evaluate),

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
