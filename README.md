# ComfyUI-KepOpenAI

## Overview
Ensure you have your OPENAI_API_KEY set as an environment variable.

ComfyUI-KepOpenAI is a user-friendly node that serves as an interface to the GPT-4 with Vision (GPT-4V) API. This integration facilitates the processing of images coupled with text prompts, leveraging the capabilities of the OpenAI API to generate text completions that are contextually relevant to the provided inputs.

## Features

- Accepts both an image and a text prompt as input.
- Integrates seamlessly with the OpenAI GPT-4V API to provide intelligent text completions.
- Requires an OpenAI API key, which should be securely stored and provided as an environment variable.

## Configuration

To utilize this node, you must set the `OPENAI_API_KEY` environment variable with your OpenAI API key. This ensures that the API is accessible and requests are authenticated properly.

## Changelog

### Version 1.1.0

- Implemented caching mechanism to store and retrieve previous API responses
- Updated default prompt to generate more detailed captions
- Increased default max_tokens from 77 to 256
- Adjusted max_tokens range to 128-2048
- Changed model from "gpt-4-vision-preview" to "gpt-4o-mini"

### Version 1.0.0

- Initial release
- Basic integration with GPT-4 Vision API
- Support for image and text prompt inputs
- Configurable max_tokens parameter
