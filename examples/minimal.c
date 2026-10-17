/* minimal.c — the smallest complete libnnc program, in plain C.
 *
 *   nnc-minimal MODEL.gguf "Why is the sky blue?"
 *
 * Loads a model, sends one message in the model's own chat format, and
 * streams the reply to stdout. Everything it uses is in nnc.h.
 */

#include "nnc.h"

#include <stdio.h>

static bool print_text(const char* text, size_t length, void* user)
{
	(void)user;
	fwrite(text, 1, length, stdout);
	fflush(stdout);
	return true; /* returning false would stop the reply */
}

int main(int argc, char** argv)
{
	nnc_model* model = NULL;
	nnc_session* session = NULL;
	nnc_generate_params gen;
	nnc_generate_result result;
	nnc_status status;

	if (argc < 3)
	{
		fprintf(stderr, "usage: %s MODEL.gguf \"message\"\n", argv[0]);
		return 2;
	}
	if (!nnc_cpu_supported())
	{
		fprintf(stderr, "this CPU lacks AVX2 and FMA, which nnc needs\n");
		return 1;
	}

	/* NULL params: the defaults (8-bit weights, no progress callback). */
	if (nnc_model_load(argv[1], NULL, &model) != NNC_OK)
	{
		fprintf(stderr, "cannot load %s: %s\n", argv[1], nnc_last_error());
		return 1;
	}
	if (nnc_session_create(model, NULL, &session) != NNC_OK)
	{
		fprintf(stderr, "cannot start a session: %s\n", nnc_last_error());
		nnc_model_free(model);
		return 1;
	}

	gen = nnc_generate_default_params();
	gen.on_text = print_text;
	status = nnc_session_chat(session, argv[2], &gen, &result);
	printf("\n");
	if (status == NNC_OK)
		fprintf(stderr, "[%d tokens at %.1f tokens/s]\n", result.n_generated,
		        result.decode_ms > 0 ? 1000.0 * result.n_generated / result.decode_ms : 0.0);
	else
		fprintf(stderr, "error: %s\n", nnc_last_error());

	nnc_session_free(session);
	nnc_model_free(model);
	return status == NNC_OK ? 0 : 1;
}
