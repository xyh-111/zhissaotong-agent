"""
重建向量库脚本（一步完成）

用途：当切换 embedding 模型（向量维度/分布可能变化）时，旧的 chroma_db 与去重记录无法复用，
需要先清空，再用新模型重新入库。

做法：
  1. 删除旧的向量库目录（config/chroma.yml -> persist_directory）
  2. 删除 MD5 去重记录（config/chroma.yml -> md5_hex_store）
  3. 用新模型重新加载 data/ 下的知识库文件

用法：
  python rebuild_vector_store.py

注意：本脚本会删除旧向量库与去重记录，但二者均可由 data/ 目录下的源文件重新生成，安全可逆。
"""
import os
import shutil

from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from rag.vector_store import VectorStoreService


def rebuild():
    persist_dir = get_abs_path(chroma_conf["persist_directory"])
    md5_store = get_abs_path(chroma_conf["md5_hex_store"])

    # 1. 清空旧的向量库目录
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        logger.info(f"[重建向量库]已删除旧向量库目录：{persist_dir}")
    else:
        logger.info(f"[重建向量库]向量库目录不存在，跳过删除：{persist_dir}")

    # 2. 清空 MD5 去重记录（否则会误以为文件已入库而跳过）
    if os.path.exists(md5_store):
        os.remove(md5_store)
        logger.info(f"[重建向量库]已删除 MD5 去重记录：{md5_store}")
    else:
        logger.info(f"[重建向量库]MD5 记录不存在，跳过删除：{md5_store}")

    # 3. 用新模型重新加载知识库
    logger.info("[重建向量库]开始重新加载知识库...")
    vs = VectorStoreService()
    vs.load_document()
    logger.info("[重建向量库]知识库重建完成")


if __name__ == "__main__":
    rebuild()
