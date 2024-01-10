from typing import Optional, Tuple

import sqlalchemy
from sqlalchemy import Column, Integer, ARRAY, String, JSON,Text
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Session, relationship

from langchain.vectorstores.pgvector import BaseModel


class CollectionStore(BaseModel):
    """Collection store."""

    __tablename__ = "langchain_pg_collection"

    name = sqlalchemy.Column(sqlalchemy.String)
    cmetadata = sqlalchemy.Column(JSON)

    embeddings = relationship(
        "EmbeddingStore",
        back_populates="collection",
        passive_deletes=True,
    )

    @classmethod
    def get_by_name(cls, session: Session, name: str) -> Optional["CollectionStore"]:
        return session.query(cls).filter(cls.name == name).first()  # type: ignore

    @classmethod
    def get_or_create(
            cls,
            session: Session,
            name: str,
            cmetadata: Optional[dict] = None,
    ) -> Tuple["CollectionStore", bool]:
        """
        Get or create a collection.
        Returns [Collection, bool] where the bool is True if the collection was created.
        """
        created = False
        collection = cls.get_by_name(session, name)
        if collection:
            return collection, created

        collection = cls(name=name, cmetadata=cmetadata)
        session.add(collection)
        session.commit()
        created = True
        return collection, created


class EmbeddingStore(BaseModel):
    """Embedding store."""

    __tablename__ = "langchain_pg_embedding"

    # 档案库名称
    archive_name = sqlalchemy.Column(sqlalchemy.String)
    # 自己添加字段-PDM文件Id
    file_id = sqlalchemy.Column(sqlalchemy.BigInteger)
    # 文件名
    file_name = sqlalchemy.Column(sqlalchemy.String)
    # 文件路径
    file_path = sqlalchemy.Column(sqlalchemy.String)
    # pdm中的地址
    pdm_path = sqlalchemy.Column(sqlalchemy.String)
    # plm中的pdm地址
    plm_pdm_path = sqlalchemy.Column(sqlalchemy.String)
    # 卡片信息
    card_info = Column(Text)
    # card_info = Column(JSON, default=JSON)
    # card_info = sqlalchemy.Column(sqlalchemy.JSON)
    # 有权限访问该条数据的用户
    # permission_users = sqlalchemy.Column(sqlalchemy.ARRAY(sqlalchemy.String))
    permission_users = Column(ARRAY(String))
    # pdm文件流程状态
    work_flow_status = sqlalchemy.Column(sqlalchemy.String)
    # 进度
    process = sqlalchemy.Column(sqlalchemy.String)
    # 版次
    rev = sqlalchemy.Column(sqlalchemy.String)
    # 修订版
    revised_edition = sqlalchemy.Column(sqlalchemy.String)
    # 工作流
    work_flow = sqlalchemy.Column(sqlalchemy.String)
    # 更新日期
    update_date = sqlalchemy.Column(sqlalchemy.String)
    # 创建人
    create_user = sqlalchemy.Column(sqlalchemy.String)
    # 创建日期
    create_date = sqlalchemy.Column(sqlalchemy.String)

    #//文件上传状态,0:默认无状态,1:上传中,2:上传成功,3:上传失败
    file_upload_process_status = Column(Integer,default=0)
    #//文件向量化状态,0:默认无状态,1:向量化中,2:向量化成功,3:向量化失败
    file_vectorization_process_status = Column(Integer,default=0)

    collection_id = sqlalchemy.Column(
        UUID(as_uuid=True),
        sqlalchemy.ForeignKey(
            f"{CollectionStore.__tablename__}.uuid",
            ondelete="CASCADE",
        ),
    )
    collection = relationship(CollectionStore, back_populates="embeddings")

    embedding: Vector = sqlalchemy.Column(Vector(None))
    document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
    cmetadata = sqlalchemy.Column(JSON, nullable=True)

    # custom_id : any user defined id
    custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)
