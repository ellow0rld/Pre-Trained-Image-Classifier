import asyncio
import instructor

import google.generativeai as genai
from pydantic import BaseModel, Field, create_model
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from scraper import webscraper
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)
DataT = TypeVar("DataT")


class BaseExtractor(BaseModel):
    name: str = Field(
        ...,
        description="Only the general name of extracted thing",
        examples=["latest_stock_details", "trending_news"],
    )
    data: DataT = Field(
        ...,
        description="The important data to be extracted, if the data is huge then it should be a list of dictionaries",
        examples=[
            {"name": "stock_name", "value": "Apple Inc."},
            {"name": "stock_price", "value": "$150.00"},
        ],
    )


class extractor:
    def __init__(
        self,
        query: str,
        url: str,
        api_key: str,
        gemini_model: str = "gemini-1.5-flash",
        fields: dict[str, Type] | None = None,
    ):
        self.query = query
        self.url = url
        self.api_key = api_key
        self.gemini_model = gemini_model
        self.fields = fields

    def __get_content(self) -> str:
        scraper = webscraper(self.url)
        try:
            content = scraper.scraping_with_langchain()
            return content
        except PlaywrightTimeoutError as pte:
            raise TimeoutError(
                "The scraping process timed out. Or the page took too long to load. Please try again later."
            )
        except Exception as e:
            raise e

    def __create_pydantic_model(self, fields: dict[str, Type]) -> Type[T]:
        data_model = create_model(
            "DataModel",
            **{
                field_name: (field_type, Field(...))
                for field_name, field_type in fields.items()
            },
        )

        dynamic_model = create_model(
            "CustomExtractor",
            name=(str, Field(..., description="Name of the item")),
            data=(list[data_model], Field(..., description="The dynamic data fields")),
        )
        return dynamic_model

    def __generate_prompt(self, content: str) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful extractor that extract and structure data.",
            },
            {
                "role": "user",
                "content": f"You will be given a content to extract information from. The content is delimited by four backticks. Also, you will be given a query of what to extract delimited by four hashtags. please have the following Query: ####{self.query}#### and here is the following Content: ```{content}```",
            },
        ]
        return messages

    def __call_gemini(
        self, prompt: list[dict], pydantic_schema: Type[T], api_key: str, gemini_model: str
    ) -> dict:
        client = instructor.from_gemini(
            client=genai.configure(api_key=api_key)
        )

        response = client.chat.completions.create(
            gemini_model=gemini_model,
            messages=prompt,
            response_model=pydantic_schema,
            temperature=0.125,
        )
        return response

    def __async_run_content(self) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        content = loop.run_until_complete(self.__get_content())
        loop.close()
        return content

    def extract(self) -> str:
        content = self.__get_content()

        pydantic_schema = (
            self.__create_pydantic_model(fields=self.fields)
            if self.fields
            else BaseExtractor
        )

        prompt = self.__generate_prompt(content)

        response = self.__call_gemini(
            prompt=prompt,
            pydantic_schema=pydantic_schema,
            api_key=self.api_key,
            gemini_model=self.gemini_model,
        )

        return response
