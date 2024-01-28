
from models.colbert import ColBERT

pretrained_model_name_or_path = "colbert-ir/colbertv2.0"
n_gpu = 1
index_root = None
verbose = True

colbert = ColBERT(
            pretrained_model_name_or_path, n_gpu, index_root=index_root, verbose=verbose
        )

documents = ['a', 'bob the builder', 'flexor ai is a great startup that will kiilll it', 'd', 'e', 'f']

queries = ['b is bob']

files = colbert.create_heatmaps_query_documents(query=queries[0],documents=documents,k = 2)
print(files)