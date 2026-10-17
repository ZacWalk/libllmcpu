/* function_call.c — a function-calling round trip with FunctionGemma, in
 * plain C.
 *
 *   nnc-function-call functiongemma-270m-it-q8_0.gguf "What's the weather in Tokyo?"
 *
 * Declares one function in the system prompt (FunctionGemma's developer
 * turn), lets the model call it, hands back a made-up result in the same
 * turn, and lets the model answer in words. Everything it uses is in nnc.h.
 */

#include "nnc.h"

#include <stdio.h>
#include <string.h>

static const char k_declarations[] =
	"You are a model that can do function calling with the following functions"
	"<start_function_declaration>declaration:get_current_weather{description:<escape>Gets the current weather "
	"in a given location.<escape>,parameters:{properties:{location:{description:<escape>The city and state, "
	"e.g. \"San Francisco, CA\" or \"Tokyo, JP\"<escape>,type:<escape>STRING<escape>}},"
	"required:[<escape>location<escape>],type:<escape>OBJECT<escape>}}<end_function_declaration>";

/* The reply so far, to look for a call in. */
typedef struct
{
	char text[4096];
	size_t length;
} reply;

static bool collect(const char* text, size_t length, void* user)
{
	reply* r = (reply*)user;
	if (r->length + length < sizeof(r->text))
	{
		memcpy(r->text + r->length, text, length);
		r->length += length;
		r->text[r->length] = '\0';
	}
	return true;
}

int main(int argc, char** argv)
{
	nnc_model* model = NULL;
	nnc_session* session = NULL;
	nnc_session_params sp;
	nnc_generate_params gen;
	nnc_token tokens[256];
	reply r;
	int32_t n;
	int rc = 1;
	/* What the function returned, in FunctionGemma's syntax. */
	const char* result = "response:get_current_weather{temperature:15,weather:<escape>sunny<escape>}"
		"<end_function_response>";

	if (argc < 3)
	{
		fprintf(stderr, "usage: %s FUNCTIONGEMMA.gguf \"question\"\n", argv[0]);
		return 2;
	}
	if (nnc_model_load(argv[1], NULL, &model) != NNC_OK)
	{
		fprintf(stderr, "cannot load %s: %s\n", argv[1], nnc_last_error());
		return 1;
	}
	sp = nnc_session_default_params();
	sp.system_prompt = k_declarations;
	if (nnc_session_create(model, &sp, &session) != NNC_OK) goto failed;

	gen = nnc_generate_default_params();
	gen.sampler.temperature = 0.0f; /* greedy, so the run is repeatable */
	gen.on_text = collect;
	gen.user = &r;

	r.length = 0;
	r.text[0] = '\0';
	if (nnc_session_chat(session, argv[2], &gen, NULL) != NNC_OK) goto failed;
	printf("model:    %s\n", r.text);
	if (!strstr(r.text, "<start_function_call>call:get_current_weather"))
	{
		printf("(the model answered without calling the function)\n");
		rc = 0;
		goto done;
	}

	/* Generation stopped where the function's response belongs. Run the
	 * function (here, pretend) and continue the same turn with its result. */
	printf("function: %s\n", result);
	n = nnc_tokenize(model, result, strlen(result), false, tokens, 256);
	if (n < 0 || n > 256) goto failed;
	r.length = 0;
	r.text[0] = '\0';
	if (nnc_session_generate(session, tokens, n, &gen, NULL) != NNC_OK) goto failed;
	printf("model:    %s\n", r.text);
	rc = 0;
	goto done;

failed:
	fprintf(stderr, "error: %s\n", nnc_last_error());
done:
	nnc_session_free(session);
	nnc_model_free(model);
	return rc;
}
