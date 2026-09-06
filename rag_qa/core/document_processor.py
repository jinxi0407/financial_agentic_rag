# core/document_processor.py
import os
import hashlib
# 文档加载器，把整个文档按照纯文本的形式加载成Document
from langchain_community.document_loaders import TextLoader
# 文档加载器，把markdown格式的数据，提取文本内容，转成Document对象
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
# 支持markdown格式的切割（根据几级标题等）
from langchain_text_splitters import MarkdownTextSplitter

from datetime import datetime

# TODO 中文递归分割器（主要因为中文和英文的标点符号不一样，所以我们不使用langchain自带的RecursiveTextSplitter）
from rag_qa.text_splitters import ChineseRecursiveTextSplitter

from rag_qa.text_splitters import AliTextSplitter

# TODO PDF、DOC、PPT、IMG 格式的数据都是我们自己写的
from rag_qa.document_loaders import OCRPDFLoader, OCRDOCLoader, OCRPPTLoader, OCRIMGLoader
from base.config import config
from base.logger import logger

# 定义文档加载器
# 1. 在遍历文档的时候知道哪种后缀的文件类型使用哪个加载器进行处理 2. 通过 字典的.keys()我们也可以知道我们的系统支持哪些文件类型
# TODO 这里的逻辑： 通过python自带的工具读取文本 + orc读取图片里的内容。 这里缺少对于表格数据的处理
document_loaders = {
    # 文本文件使用 TextLoader
    ".txt": TextLoader,
    # PDF 文件使用 OCRPDFLoader
    ".pdf": OCRPDFLoader,
    # Word 文件使用 OCRDOCLoader
    ".docx": OCRDOCLoader,
    # PPT 文件使用 OCRPPTLoader
    ".ppt": OCRPPTLoader,
    # PPTX 文件使用 OCRPPTLoader
    ".pptx": OCRPPTLoader,
    # JPG 文件使用 OCRIMGLoader
    ".jpg": OCRIMGLoader,
    # PNG 文件使用 OCRIMGLoader
    ".png": OCRIMGLoader,
    # Markdown 文件使用 UnstructuredMarkdownLoader
    ".md": UnstructuredMarkdownLoader
}

"""
需求：从指定文件夹加载多种类型文件并添加元数据
思路：
1. 遍历目录下的文件
2. 过滤文件类型，并加载文件
    2.1 基于文件类型构造加载器类型。如果是txt，需要指定编码为utf-8。 
    2.2 加载对应的文件并转为Document
3. 给每个文档添加元数据：知识库来源、路径、时间戳
"""


def calculate_file_sha256(file_path, block_size=1024 * 1024):
    """Return the stable content identity used for one source file."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as source_file:
        for block in iter(lambda: source_file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parent_id(document_id, parent_index):
    return f"{document_id}_parent_{parent_index}"


def build_child_id(document_id, parent_index, child_index):
    return f"{build_parent_id(document_id, parent_index)}_child_{child_index}"


def build_milvus_primary_key(document_id, parent_index, child_index, child_text):
    """Build a stable key that remains unique across files with identical text."""
    identity = f"{document_id}\0{parent_index}\0{child_index}\0{child_text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_documents_from_directory(directory_path, is_document_ingested=None):
    """
    从指定文件夹加载多种类型文件并添加元数据，把处理文档返回，用于后续写入向量库
    :param directory_path: 要读取的目录（里面存放着需要被加载到向量库的所有文档）
    :return: 所有加载好并处理成Document类型(转成文本、 添加元数据 写入向量库的时候使用)的集合（列表）
    """

    # 定义一个空的list，用于存放需要返回的结果
    documents = []

    # 拿到所有的支持的文件类型
    supported_extensions = document_loaders.keys()

    # ai_data -? ai
    # 目录名会作为 source，例如 annual_reports。
    source = os.path.basename(directory_path).replace("_data", "")

    # os.walk： 递归遍历 directory_path 下面的所有文件， 一次循环处理一个目录
    # root：当前遍历目录的路径
    #  _: 当前目录下面有哪些文件夹
    # files: 当前目录下面有哪些文件名，例如 annual_report.pdf

    for root, directories, files in os.walk(directory_path):
        directories.sort()
        files.sort()
        for file in files:
            # 当前目录路径 + 文件名 = 当前目录下面所有文件的完整路径
            file_path = os.path.join(root, file)
            # 获得当前处理的这个文件的后缀 (.txt / .pdf)
            extension_name = os.path.splitext(file_path)[1].lower()
            # 判断文件类型是否够是我们支持的
            if extension_name in supported_extensions:
                try:
                    file_sha256 = calculate_file_sha256(file_path)
                    document_id = file_sha256
                    if is_document_ingested and is_document_ingested(document_id):
                        logger.info("文件已入库，跳过处理: %s", file_path)
                        continue

                    # 根据后缀名，TODO 拿到文件加载器类  ，构造方法： 类名()
                    loader_class = document_loaders[extension_name]
                    # TODO 构建文件加载器对象
                    # OCRPDFLoader()
                    if extension_name == '.txt':
                        loader = loader_class(file_path, encoding="utf-8")
                    else:
                        loader = loader_class(file_path)

                    # TODO 上述代码等同于
                    # if extension_name == '.txt':
                    #     loader = TextLoader(file_path, encoding="utf-8")
                    # elif extension_name == '.pdf':
                    #     loader = OCRPDFLoader(file_path)
                    # elif extension_name == '.docx':
                    #     loader = OCRDOCLoader(file_path)

                    # 返回的就是加载器加载好的完整的文档
                    # list[Document[ metadata(dict结构), page_content:大字符串 ] ]
                    # 里面只有一个元素， 就相当于 一个pdf/doc/ppt文件 -> Document对象
                    loaded_docs = loader.load()

                    # 添加知识库来源、文件路径、后缀名、时间等元数据，用于写入向量库
                    # 这里面实际就执行了一次，因为list里面只有一个元素
                    for doc in loaded_docs:
                        doc.metadata['source'] = source
                        doc.metadata['file_path'] = file_path
                        doc.metadata['extension'] = extension_name
                        doc.metadata['document_id'] = document_id
                        doc.metadata['file_sha256'] = file_sha256
                        doc.metadata['source_filename'] = file
                        doc.metadata['timestamp'] = datetime.now().isoformat()

                    # extend方法：  loaded_docs: list[Document]
                    # documents.append : [ [doc1, doc2] ,[doc3, doc4]]
                    # documents.extend :  [ doc1, doc2 ,doc3, doc4]. 把loaded_docs拆开，所有元素放入documents中
                    documents.extend(loaded_docs)

                    logger.info(f'成功加载文件: {file_path}')
                except Exception as e:
                    logger.error(f'加载文件失败:{file_path}, error:{e}')
            else:
                logger.warning(f'不支持的文件类型:{file_path}')
    return documents


"""
需求：处理文档并进行分层切分，返回子块结果
思路：
1. 获取所有的Document
2. 初始化文档分割器
    2.1 对于markdown格式，使用MarkdownTextSplitter
    2.2 对于其他格式，使用ChineseRecursiveTextSplitter
3. 把文件分割成多个子块
    3.1 把一个文档分成多个块chunk
    3.2 把每一个子块近一步切分成子块sub_chunk
    3.3 给每个子块添加元数据，包括： ①parent_id ②id ③parent_content
"""


# 定义函数，处理文档并进行分层切分，返回子块结果
def process_documents(directory_path,
                      parent_chunk_size=config.PARENT_CHUNK_SIZE,
                      child_chunk_size=config.CHILD_CHUNK_SIZE,
                      chunk_overlap=config.CHUNK_OVERLAP,
                      is_document_ingested=None):
    """
    处理文档并进行分层切分，返回子块结果
    :param directory_path:      要处理的文档路径
    :param parent_chunk_size:   父块大小
    :param child_chunk_size:    子块大小
    :param chunk_overlap:       重叠字符数
    :return: 所有的子块
    """
    # 从指定目录加载所有文档
    # list[Document[metadata(dict，自定义),page_content=内容]]
    documents = load_documents_from_directory(
        directory_path,
        is_document_ingested=is_document_ingested,
    )

    # 记录加载的文档总数日志
    logger.info(f"加载的文档数量: {len(documents)}")

    # 初始化父块和子块分词器（通用）
    parent_splitter = ChineseRecursiveTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    child_splitter = ChineseRecursiveTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)
    # 初始化 Markdown 专用分词器
    markdown_parent_splitter = MarkdownTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=chunk_overlap)
    markdown_child_splitter = MarkdownTextSplitter(chunk_size=child_chunk_size, chunk_overlap=chunk_overlap)

    # 初始化空列表，用于存储所有子块
    child_chunks = []

    # Parent indexes are scoped to a stable source-file document_id.
    parent_indexes = {}

    # 遍历每个原始文档，带上索引 i
    # TODO 第一次分割：分父块
    # enumerate ： 在遍历一个list等可迭代元素同时，添加当前遍历的是第几个元素
    for doc in documents:
        # 获取文件扩展名
        file_extension = os.path.splitext(doc.metadata.get("file_path", ""))[1].lower()

        # 选择切分器
        is_markdown = (file_extension == ".md")
        parent_splitter_to_use = markdown_parent_splitter if is_markdown else parent_splitter
        # print(f'parent_splitter_to_use-->{parent_splitter_to_use}')
        child_splitter_to_use = markdown_child_splitter if is_markdown else child_splitter

        logger.info(
            f"处理文档: {doc.metadata['file_path']}, 使用切分器: {'Markdown' if is_markdown else 'ChineseRecursive'}")

        document_id = doc.metadata["document_id"]

        # 使用父块分词器将文档切分为父块
        # 切割器需要传入的是list格式，需要转一个转换 Document -> [Document]

        # TODO parent_docs -> list[Document] 每个Document对应的一个父块
        parent_docs = parent_splitter_to_use.split_documents([doc])
        # 遍历每个父块，带上索引 j
        # TODO 第二次分割：分子块
        for parent_doc in parent_docs:
            parent_index = parent_indexes.get(document_id, 0)
            parent_indexes[document_id] = parent_index + 1

            # Parent identity is stable across runs and independent of traversal order.
            parent_id = build_parent_id(document_id, parent_index)
            # 将父块 ID 添加到元数据
            parent_doc.metadata["parent_id"] = parent_id

            # 使用子块分词器将父块切分为子块
            sub_chunks = child_splitter_to_use.split_documents([parent_doc])

            # 遍历每个子块，带上索引 k
            for k, sub_chunk in enumerate(sub_chunks):
                # 为子块添加父块 ID 到元数据
                sub_chunk.metadata["parent_id"] = parent_id
                # 为子块添加父块内容到元数据
                # TODO 把父块的内容添加到子块的元数据，是为了在向量库检索的时候直接通过子块的匹配拿到父块的全部文本内容
                sub_chunk.metadata["parent_content"] = parent_doc.page_content
                # Keep both the logical child ID and the Milvus primary key explicit.
                sub_chunk.metadata["id"] = build_child_id(document_id, parent_index, k)
                sub_chunk.metadata["milvus_id"] = build_milvus_primary_key(
                    document_id, parent_index, k, sub_chunk.page_content
                )
                # 将子块添加到子块列表中
                child_chunks.append(sub_chunk)

    # 记录子块总数日志
    logger.info(f"子块数量: {len(child_chunks)}")
    # 返回所有子块列表
    return child_chunks


if __name__ == '__main__':
    child_chunks = process_documents(
        os.path.join(config.FINANCIAL_DATA_DIR, 'annual_reports'))
    print(child_chunks)
