/* embed.c — semantic search with a text-embedding model, in plain C.
 *
 *   nnc-embed embeddinggemma-300M-Q8_0.gguf "Which planet is known as the Red Planet?"
 *
 * Embeds a few documents and the query, then ranks the documents by how
 * close their meaning is to the query's. Embeddings have unit length, so
 * the dot product of two is their cosine similarity. Everything it uses is
 * in nnc.h.
 */

#include "nnc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* const k_documents[] = {
	"Venus is often called Earth's twin because of its similar size and proximity.",
	"Mars, known for its reddish appearance, is often referred to as the Red Planet.",
	"Jupiter, the largest planet in our solar system, has a prominent red spot.",
	"Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
};
#define N_DOCS ((int)(sizeof(k_documents) / sizeof(k_documents[0])))

/* EmbeddingGemma wants the task in front of the text: one prefix for
 * search queries, another for the documents they are matched against. */
static bool embed(const nnc_model* model, const char* prefix, const char* text, float* out, int n)
{
	const size_t a = strlen(prefix), b = strlen(text);
	char* full = (char*)malloc(a + b);
	bool ok;
	if (!full) return false;
	memcpy(full, prefix, a);
	memcpy(full + a, text, b);
	ok = nnc_embed(model, full, a + b, out, n) == NNC_OK;
	free(full);
	return ok;
}

int main(int argc, char** argv)
{
	nnc_model* model = NULL;
	nnc_model_info info;
	float* query = NULL;
	float* doc = NULL;
	float score[N_DOCS];
	int order[N_DOCS];
	int i, j, n, rc = 1;

	if (argc < 3)
	{
		fprintf(stderr, "usage: %s EMBEDDING-MODEL.gguf \"query\"\n", argv[0]);
		return 2;
	}
	if (nnc_model_load(argv[1], NULL, &model) != NNC_OK)
	{
		fprintf(stderr, "cannot load %s: %s\n", argv[1], nnc_last_error());
		return 1;
	}
	info = nnc_model_get_info(model);
	n = info.n_embd_out;
	if (n == 0)
	{
		fprintf(stderr, "%s is not a text-embedding model\n", info.name);
		goto done;
	}
	query = (float*)malloc(sizeof(float) * (size_t)n);
	doc = (float*)malloc(sizeof(float) * (size_t)n);
	if (!query || !doc) goto done;

	if (!embed(model, "task: search result | query: ", argv[2], query, n)) goto failed;
	for (i = 0; i < N_DOCS; ++i)
	{
		float dot = 0.0f;
		if (!embed(model, "title: none | text: ", k_documents[i], doc, n)) goto failed;
		for (j = 0; j < n; ++j) dot += query[j] * doc[j];
		score[i] = dot;
		order[i] = i;
	}

	/* Best match first. */
	for (i = 1; i < N_DOCS; ++i)
		for (j = i; j > 0 && score[order[j]] > score[order[j - 1]]; --j)
		{
			const int t = order[j];
			order[j] = order[j - 1];
			order[j - 1] = t;
		}
	for (i = 0; i < N_DOCS; ++i) printf("%.3f  %s\n", score[order[i]], k_documents[order[i]]);
	rc = 0;
	goto done;

failed:
	fprintf(stderr, "error: %s\n", nnc_last_error());
done:
	free(query);
	free(doc);
	nnc_model_free(model);
	return rc;
}
