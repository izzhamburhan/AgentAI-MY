import os 
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader


# def get_index(data, index_name):
#     index = None 
#     if not os.path.exists(index_name):
#         print('Building index', index_name)
#         index = VectorStoreIndex.from_documents(data, show_progress=True)
#         index.storage_context.persist(persist_dir=index_name)
#     else :
#         index = load_index_from_storage(
#             StorageContext.from_defaults(persist_dir=index_name)
#             )
    
#     return index

# pdf_path = os.path.join('data', 'Malaysia.pdf')
# malaysia_pdf = PDFReader().load_data(file=pdf_path)
# malaysia_index = get_index(malaysia_pdf, 'malaysia')
# malaysia_engine = malaysia_index.as_query_engine()
# malaysia_engine.query()

def get_index(data_files, index_name):
    index = None
    data = []
    for file_path in data_files:
        reader = PDFReader()
        data.extend(reader.load_data(file=file_path))

    if not os.path.exists(index_name):
        print(f'Building index {index_name}')
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index

file_paths = [
    os.path.join('data', 'Malaysia.pdf'),
    # Add more file paths here
]
combined_index = get_index(file_paths, 'combined_index')
combined_engine = combined_index.as_query_engine()