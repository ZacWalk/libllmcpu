// nnc — tokenizer: vocabulary, encoding, decoding and chat-template tokens.
//
// Three encoders, chosen by `tokenizer.ggml.model`:
//   "llama"  SentencePiece BPE driven by per-token scores (Gemma 3, Llama 2),
//            with <0xHH> byte fallback;
//   "gpt2"   byte-level BPE driven by merge ranks (Llama 3, Qwen), with a
//            pre-tokenizer that splits words, numbers and punctuation the
//            way the llama-bpe regex does (Qwen's splits digits singly);
//   other    BPE over ▁-for-space codepoints driven by merge ranks (Gemma 4).
// Decoding renders byte tokens as their byte and control tokens as
// nothing, so streamed text never shows "<0x0A>" or "<end_of_turn>".

#pragma once

#include "gguf.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// GGUF tokenizer.ggml.token_type values.
enum vocab_token_type : int32_t
{
	VOCAB_TOKEN_NORMAL = 1,
	VOCAB_TOKEN_UNKNOWN = 2,
	VOCAB_TOKEN_CONTROL = 3,
	VOCAB_TOKEN_USER_DEFINED = 4,
	VOCAB_TOKEN_UNUSED = 5,
	VOCAB_TOKEN_BYTE = 6,
};

enum class vocab_kind
{
	spm, // "llama": SentencePiece scores
	gpt2, // byte-level BPE
	bpe, // ▁-for-space BPE by merge rank
};

// How a byte-level BPE vocabulary cuts text before merging
// (tokenizer.ggml.pre). The regexes differ in their digit runs and in
// whether combining marks join the letters around them.
enum class vocab_pre
{
	llama3, // "llama-bpe" and anything unrecognised: up to three digits per piece
	qwen2, // "qwen2": one digit per piece
	qwen35, // "qwen35": one digit per piece, and combining marks count as letters
};

// Which turn format nnc_session_chat() writes.
enum class chat_format
{
	none, // unknown: user text goes in as it is
	gemma, // <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n
	gemma4, // <|turn>user\n...<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
	llama3, // <|start_header_id|>user<|end_header_id|>\n\n...<|eot_id|>
	chatml, // <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n (Qwen)
};

struct model_vocab
{
	vocab_kind kind = vocab_kind::spm;
	vocab_pre pre = vocab_pre::llama3;
	std::string model; // tokenizer.ggml.model, as written in the file

	std::vector<std::string> tokens; // as the file spells them ("▁Paris", "<0x0A>")
	std::vector<float> scores;
	std::vector<int32_t> types; // vocab_token_type per token (may be empty)
	std::vector<std::string> pieces; // the bytes each token renders to

	std::unordered_map<std::string, int> token_to_id;
	std::unordered_map<std::string, int> merge_rank; // "lhs rhs" -> rank

	int bos_id = -1, eos_id = -1, unk_id = -1, pad_id = -1;
	bool add_bos = true;
	bool add_eos = false; // end every sequence with EOS (embedding models)
	bool add_space_prefix = false;
	int byte_ids[256]; // <0xHH> token per byte, or -1

	std::vector<int> eog_ids; // end-of-generation: EOS and end-of-turn tokens

	// User-defined tokens (function-call tags, runs of spaces or newlines):
	// always one token wherever their text appears in the input. Sorted by
	// first byte, then longest first; those starting with byte b are
	// user_defined[user_defined_at[b] .. user_defined_at[b + 1]).
	std::vector<std::pair<std::string, int>> user_defined;
	uint32_t user_defined_at[257] = {};

	chat_format chat = chat_format::none;
	int tok_turn_start = -1; // <start_of_turn> | <|turn> | <|start_header_id|> | <|im_start|>
	int tok_turn_end = -1; // <end_of_turn> | <turn|> | <|eot_id|> | <|im_end|>
	int tok_header_end = -1; // <|end_header_id|> (llama3)
	int tok_newline = -1; // "\n" (gemma)
	int tok_channel_start = -1; // <|channel> (gemma4): opens the thought channel
	int tok_channel_end = -1; // <channel|>
	// The file's chat template opens each reply with an empty thought
	// channel (Gemma 4 26B/31B with thinking off; E2B/E4B do not).
	bool empty_thought = false;
	// FunctionGemma: the system prompt (with the function declarations)
	// is a turn of its own, under the "developer" role.
	bool developer_turn = false;
	// ChatML, from the file's template: the system prompt it writes when
	// none is given (Qwen2.5's), and how it opens each reply - with an empty
	// <think> block when thinking can be turned off (Qwen3, 3.5, 3.8), or
	// with "<think>\n" for models that always think.
	std::string default_system;
	std::string reply_open;
	// Thinking. can_think: nnc_session_params.thinking lets the model reason
	// before it answers (Gemma 4, whose template has <|think|>; ChatML
	// templates with an enable_thinking switch). always_thinks: it reasons
	// whatever it is told. With thinking on, a ChatML reply opens with
	// reply_open_think ("<think>\n", or nothing for Qwen3, which opens the
	// block itself), and the system turn starts with think_instructions
	// (Qwen3.8's default reasoning effort); Gemma 4's system turn starts
	// with tok_think.
	bool can_think = false;
	bool always_thinks = false;
	std::string reply_open_think;
	std::string think_instructions;
	int tok_think = -1; // <|think|> (Gemma 4)
	// The markers around the model's reasoning, which tell it apart from the
	// answer: <think> and </think>, or Gemma 4's thought channel, whose first
	// line is the channel's name (thought_label). -1 without them.
	int tok_thought_open = -1;
	int tok_thought_close = -1;
	bool thought_label = false;
	// TranslateGemma: language code ("en", "pt-BR") -> English name, from
	// the chat template. Empty for other models.
	std::unordered_map<std::string, std::string> languages;

	int n_vocab() const { return static_cast<int>(tokens.size()); }
	bool valid(const int id) const { return id >= 0 && id < n_vocab(); }
};

// Read the tokenizer.ggml.* metadata. On failure records the reason with
// nnc_fail() and returns false.
bool vocab_load(const gguf_file& f, model_vocab& v);

// Encode UTF-8 text. Control-token spellings in `text` stay plain text;
// user-defined tokens are matched whole, as llama.cpp and Hugging Face do.
std::vector<int> vocab_tokenize(const model_vocab& v, const std::string& text, bool add_bos);

// How a byte-level BPE vocabulary's pre-tokenizer cuts `text` (which holds
// no user-defined token): the pieces whose bytes are merged separately.
std::vector<std::string> vocab_pretokenize(const model_vocab& v, const std::string& text);

// The bytes `token` renders to ("" for control tokens and invalid ids).
const std::string& vocab_piece(const model_vocab& v, int token);

std::string vocab_detokenize(const model_vocab& v, const int* tokens, size_t n);

bool vocab_is_eog(const model_vocab& v, int token);
