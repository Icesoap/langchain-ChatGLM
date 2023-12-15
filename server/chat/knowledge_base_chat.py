import asyncio
import json
import os
from typing import AsyncIterable, List, Optional
from urllib.parse import urlencode

from fastapi import Body
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate
from sse_starlette import EventSourceResponse

from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE)
from server.chat.utils import History
from server.knowledge_base.kb_doc_api import search_docs, search_docs_custom
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.utils import BaseResponse, get_prompt_template, BaseResponseSSE
from server.utils import wrap_done, get_ChatOpenAI


# 自己添加的方法-与知识库对话接口 只查询知识库,用户前端快速搜索
async def knowledge_base_chat_only(query: str = Body(..., description="用户输入", examples=["你好"]),
                                   knowledge_base_name: str = Body(..., description="知识库名称",
                                                                   examples=["samples"]),
                                   user_name: str = Body(None, description="用户名,用于PDM判断用户权限",
                                                         examples=["用户1"]),
                                   top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                                   score_threshold: float = Body(SCORE_THRESHOLD,
                                                                 description="知识库匹配相关度阈值，取值范围在0-2之间，SCORE越小,距离越近，相关度越高，取到2相当于无门槛，建议设置在0.5左右",
                                                                 ge=0, le=5),
                                   history: List[History] = Body([],
                                                                 description="历史对话",
                                                                 examples=[[
                                                                     {"role": "user",
                                                                      "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                     {"role": "assistant",
                                                                      "content": "虎头虎脑"}]]
                                                                 ),
                                   stream: bool = Body(True, description="流式输出"),
                                   model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                                   temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                                   max_tokens: int = Body(None,
                                                          description="限制LLM生成Token数量，默认None代表模型最大值"),
                                   prompt_name: str = Body("default",
                                                           description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                                   ):
    print(
        f"query:{query},knowledge_base_name:{knowledge_base_name},user_name:{user_name},score_threshold:{score_threshold}"
        f",history:{history},stream:{stream},model_name:{model_name},temperature:{temperature},prompt_name:{prompt_name}")

    # 设置字段默认值
    # prompt_name = "knowledge_first"
    score_threshold = 5
    # temperature = 0.7

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
        # 使用自己的SSE报错类
        # base_response_sse = BaseResponseSSE()
        # return EventSourceResponse(base_response_sse.stream_err(msg=f"未找到知识库:{knowledge_base_name}"))

    # history = [History.from_data(h) for h in history]

    embedding_filter = None
    if user_name and user_name != '-1':
        embedding_filter = {"permission_users": user_name}

    # 搜索知识库
    docs = search_docs_custom(query, knowledge_base_name, top_k, score_threshold, embedding_filter)
    return BaseResponse(code=200, msg="取值成功", data=docs)


# 自己添加的方法-与知识库对话接口,主要查PDM(有权限过滤)
# 通过prompt可以设置 1查询知识库和LLM 2.只查询知识库
async def knowledge_base_chat_custom(query: str = Body(..., description="用户输入", examples=["你好"]),
                                     knowledge_base_name: str = Body(..., description="知识库名称",
                                                                     examples=["samples"]),
                                     user_name: str = Body(None, description="用户名",
                                                           examples=["用户1"]),
                                     top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                                     score_threshold: float = Body(SCORE_THRESHOLD,
                                                                   description="知识库匹配相关度阈值，取值范围在0-5之间，SCORE越小，相关度越高，取到5相当于无门槛，建议设置在0.5左右",
                                                                   ge=0, le=5),
                                     history: List[History] = Body([],
                                                                   description="历史对话",
                                                                   examples=[[
                                                                       {"role": "user",
                                                                        "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                       {"role": "assistant",
                                                                        "content": "虎头虎脑"}]]
                                                                   ),
                                     stream: bool = Body(True, description="流式输出"),
                                     model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                                     temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                                     max_tokens: int = Body(None,
                                                            description="限制LLM生成Token数量，默认None代表模型最大值"),
                                     prompt_name: str = Body("knowledge_first",
                                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                                     query_to: int = Body(0,
                                                          description="表示要从哪里查询,0:pdm;1:智能客服")
                                     ):
    print(
        f"query:{query},knowledge_base_name:{knowledge_base_name},user_name:{user_name},score_threshold:{score_threshold}"
        f",history:{history},stream:{stream},model_name:{model_name},temperature:{temperature}"
        f",prompt_name:{prompt_name},query_to:{query_to}")

    # 设置字段默认值
    # prompt_name = "knowledge_first"
    if prompt_name == "knowledge_first":
        score_threshold = 1.5

    # temperature = 0.7

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        # return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
        # 使用自己的SSE报错类
        base_response_sse = BaseResponseSSE()
        return EventSourceResponse(base_response_sse.stream_err(msg=f"未找到知识库:{knowledge_base_name}"))

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator_custom(query: str,
                                                  top_k: int,
                                                  history: Optional[List[History]],
                                                  model_name: str = LLM_MODEL,
                                                  prompt_name: str = prompt_name,
                                                  ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        embedding_filter = None
        if user_name and user_name != '-1':
            embedding_filter = {"permission_users": user_name}
        # 搜索知识库
        docs = search_docs_custom(query, knowledge_base_name, top_k, score_threshold, embedding_filter)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            url = f"/knowledge_base/download_doc?" + parameters
            if query_to == 0:
                if 'pdm_path' in doc.metadata.keys():
                    url = doc.metadata["pdm_path"]
                url = url + '```'
                # 自己添加的属性 增加plm_pdm_path
                if 'plm_pdm_path' in doc.metadata.keys():
                    url = url + doc.metadata["plm_pdm_path"]

            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)
            print("begin")
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                # yield f'data:{json.dumps({"answer": token}, ensure_ascii=False)}'
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        await task

    # result = StreamingResponse(content=knowledge_base_chat_iterator(query=query,
    #                                                                 top_k=top_k,
    #                                                                 history=history,
    #                                                                 model_name=model_name,
    #                                                                 prompt_name=prompt_name),
    #                            media_type="text/event-stream")
    result = EventSourceResponse(content=knowledge_base_chat_iterator_custom(query=query,
                                                                             top_k=top_k,
                                                                             history=history,
                                                                             model_name=model_name,
                                                                             prompt_name=prompt_name),
                                 media_type="text/event-stream")

    return result


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(SCORE_THRESHOLD,
                                                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                            ge=0, le=2),
                              history: List[History] = Body([],
                                                            description="历史对话",
                                                            examples=[[
                                                                {"role": "user",
                                                                 "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                {"role": "assistant",
                                                                 "content": "虎头虎脑"}]]
                                                            ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: int = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
                              prompt_name: str = Body("default",
                                                      description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(query: str,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODEL,
                                           prompt_name: str = prompt_name,
                                           ) -> AsyncIterable[str]:
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            url = f"/knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)
        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        await task

    result = StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                            top_k=top_k,
                                                            history=history,
                                                            model_name=model_name,
                                                            prompt_name=prompt_name),
                               media_type="text/event-stream")

    return result
