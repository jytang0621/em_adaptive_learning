# -*- coding: utf-8 -*-
# @Author  : 
# @Desc    :
import asyncio
from dis import Instruction
import os
from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI, AsyncStream, APIConnectionError
from metagpt.logs import logger

class LLMProvider:
    """处理与LLM API通信的工具类"""
    
    def __init__(self, api_key: Optional[str] = None, organization: Optional[str] = None,
                 base_url: Optional[str] = "https://openrouter.ai/api/v1"):
        """
        初始化LLM提供者

        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量OPENAI_API_KEY获取
            organization: OpenAI组织ID，如果为None则从环境变量OPENAI_ORGANIZATION获取
            base_url: API基础URL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization or os.environ.get("OPENAI_ORGANIZATION")
        
        if not self.api_key:
            raise ValueError("API密钥未提供，请设置OPENAI_API_KEY环境变量或在初始化时提供")
        
        # 确保 base_url 以 /v1 结尾（OpenAI 兼容 API 需要）
        if base_url and not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
            if base_url.endswith('/'):
                base_url = base_url + 'v1'
            else:
                base_url = base_url + '/v1'
        
        self.base_url = base_url
        self.aclient = AsyncOpenAI(api_key=self.api_key, base_url=base_url)
    
    def _cons_kwargs(self, messages: List[Dict[str, str]], timeout: int = 300, **extra_kwargs) -> Dict[str, Any]:
        """
        构建API调用参数

        Args:
            messages: 消息列表
            timeout: 超时时间（秒）
            extra_kwargs: 额外参数

        Returns:
            构建好的参数字典
        """
        kwargs = {
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.3,
            "model": "gpt-4o",
            "timeout": timeout,
            "stop": ["<|end_of_text|>", "<|eot_id|>"],
        }
        
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return kwargs


    async def run(self, model, messages: List[Dict[str, str]], use_stream = True):
        kwargs = {
            "messages": messages,
            "max_tokens": 6000,
            "temperature": 0.3,
            "model": model,
            "timeout": 600,
            "stop": ["<|end_of_text|>", "<|eot_id|>"],
        }
        
        
        try:
            response = await self.aclient.chat.completions.create(
                **self._cons_kwargs(
                    **kwargs
                ),
                stream=True
            )
        
            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""
                collected_messages.append(chunk_message)
        
            full_reply_content = "".join(collected_messages)
            return full_reply_content
    
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            raise

    async def evaluate_image(self, image_url, query, user_prompt, **kwargs):
        user_prompt = user_prompt.format(instruction=query)
        return await self.run(image_data_url=image_url, user_prompt=user_prompt, **kwargs)
    
    async def generate_reflection(self, reflection_thought, **kwargs):
        reflection_thought = reflection_thought.replace('When using the Tell or Wait action, there is no need to do reflection.', '')
        prompt = '''
You are a "precise GUI test adjudicator."

Task:
Decide if an action succeeded based only on the Reflection text below. If the Reflection is empty, whitespace-only, or contains no meaningful information, the outcome is failure.

Input:
{reflection_thought}

Decision rules (strict):
- Success (output Yes): The Reflection clearly and explicitly states that the intended UI change or functional effect occurred, e.g.:
  - "Search results updated/refreshed"
  - "Modal opened/closed"
  - "Button became enabled/disabled"
  - "Status 200 and content refreshed/list updated"
  - "Navigated to target page / expected element is visible"
- Failure (output No): Any of the following:
  - Page unresponsive, stuck, or loading never completes (e.g., "no change," "spinner keeps spinning," "timeout," "no results returned")
  - Result not as expected (e.g., "navigated to wrong page," "content differs from expectation," "button still disabled," "filter did not apply," "still old data/same page")
  - Errors/exceptions/stack traces/not implemented (e.g., "4xx/5xx error," "threw exception," "not implemented")
  - Only browser default behavior is triggered (e.g., "right-click shows browser menu only")
  - Ambiguous, contradictory, or unverifiable statements (be conservative: treat as failure)
- Negative examples (always failure): "No visible change," "not sure if it worked," "might have succeeded," "seems updated but no new data," "loading not finished."

Output format:
Return exactly one line of JSON with no extra keys:

```json
{{ "result" : "Yes" }}
```

or

```json
{{ "result" : "No" }}
```

'''
        prompt = prompt.format(reflection_thought=reflection_thought)
        # logger.info(f"prompt: {prompt}")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return await self.run(model=kwargs["model"], messages=messages)

if __name__ == "__main__":
    api_key = "sk-ekJiZhzXC9p1wQG7mXfydlazV7LTqiurBI5Hr5W2l5X3gWaE"
    api_key = "sk-zCoMKnMGAOqWuc55D0eB4vxIx8JQEA7xgetEqTHfXkchv9tk"
    base_url = "https://newapi.deepwisdom.ai/v1"
    
    model = "anthropic/claude-sonnet-4.5"
    model = "claude-sonnet-4-20250514"
    # model = "gemini-3-pro-preview"
    llm = LLMProvider(api_key=api_key, base_url=base_url)
    reflection_thought = "Generate a web application for weather app"
    # reflection_thought = 'Comparing the before and after screenshots: 1. The operation was to click the "Reload" button at coordinates (960, 716) 2. Looking at both screenshots, they appear identical with the same elements and layout:    - Same error message "Your app is not ready yet"    - Same button and text positions    - Same URL in the address bar 3. The reload action did not produce any visible changes to the page content 4. This suggests either:    - The page is still loading (though 5 seconds wait time should be'
    result = asyncio.run(llm.generate_reflection(reflection_thought=reflection_thought, model=model))
    print(result)