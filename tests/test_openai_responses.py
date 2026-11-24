"""Offline Responses API regressions using the real SDK and an HTTP stub.

Run with the project's runtime dependencies installed:
    python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
import openai
from PIL import Image

import autofigure2


def error_response(param="temperature", code=None, message=None, status=400):
    return httpx.Response(status, json={"error": {
        "code": code,
        "message": message or "Unsupported parameter: 'temperature' is not supported with this model.",
        "param": param,
        "type": "invalid_request_error",
    }})


def svg_response():
    return httpx.Response(200, json={
        "id": "resp_test", "object": "response", "created_at": 0,
        "model": "test-responses-model", "status": "completed", "error": None,
        "incomplete_details": None, "instructions": None,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                    "text": "<svg/>", "annotations": []}]}],
    })


class ResponsesTemperatureTests(unittest.TestCase):
    def invoke(self, multimodal, replies, expected_error=None):
        requests = []

        def respond(request):
            self.assertEqual(request.url.path, "/v1/responses")
            requests.append(json.loads(request.content))
            self.assertLessEqual(len(requests), len(replies), "Unexpected extra request")
            return replies[len(requests) - 1]

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            with openai.OpenAI(api_key="test-key", base_url="https://example.test/v1",
                               http_client=http_client, max_retries=0) as client:
                with patch("openai.OpenAI", return_value=client), contextlib.redirect_stdout(io.StringIO()):
                    args = dict(api_key="test-key", model="test-responses-model",
                                base_url="https://example.test/v1", provider="openai_response",
                                max_tokens=1234, temperature=0.25)

                    def call():
                        if multimodal:
                            return autofigure2.call_llm_multimodal(
                                ["Draw SVG", Image.new("RGB", (2, 2))], **args)
                        return autofigure2.call_llm_text("Draw SVG", **args)

                    if expected_error:
                        with self.assertRaises(expected_error) as raised:
                            call()
                        result = raised.exception
                    else:
                        try:
                            result = call()
                        except openai.APIError as error:
                            self.fail(f"Expected SVG output, got {error}")
                        self.assertEqual(result, "<svg/>")

        first = requests[0]
        self.assertEqual(first["model"], "test-responses-model")
        self.assertEqual(first["max_output_tokens"], 1234)
        self.assertEqual(first["temperature"], 0.25)
        self.assertEqual(first["input"][0]["role"], "user")
        content = first["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "Draw SVG"})
        self.assertEqual(len(content), 2 if multimodal else 1)
        if multimodal:
            self.assertEqual(content[1]["type"], "input_image")
            self.assertEqual(content[1]["detail"], "high")
            self.assertTrue(content[1]["image_url"].startswith("data:image/png;base64,"))
        return result, requests

    def assert_fallback_payload(self, requests):
        self.assertEqual(len(requests), 2)
        expected = dict(requests[0])
        del expected["temperature"]
        self.assertEqual(requests[1], expected)

    def test_supported_models_keep_temperature_without_retry(self):
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                _, requests = self.invoke(multimodal, [svg_response()])
                self.assertEqual(len(requests), 1)

    def test_unsupported_temperature_retries_without_changing_other_fields(self):
        for multimodal in (False, True):
            for code in (None, "unsupported_parameter"):
                with self.subTest(multimodal=multimodal, code=code):
                    # The observed error has no code; structured codes should
                    # also work without relying on the server's message text.
                    message = "Unsupported request setting." if code else None
                    _, requests = self.invoke(multimodal, [
                        error_response(code=code, message=message), svg_response(),
                    ])
                    self.assert_fallback_payload(requests)

    def test_other_bad_requests_propagate_without_retry(self):
        cases = [
            ("max_output_tokens", "unsupported_parameter", "Unsupported parameter: max_output_tokens"),
            ("temperature", "invalid_value", "Invalid temperature: must be between 0 and 2."),
            ("temperature", None, "Invalid temperature configuration."),
            ("temperature", None, "Unsupported parameter value: temperature must be between 0 and 2."),
            ("temperature", "invalid_value", "Unsupported parameter value: temperature must be between 0 and 2."),
            ("temperature", "invalid_value", "Unsupported parameter: invalid temperature value."),
            (None, None, "Unsupported parameter: 'temperature' is not supported with this model."),
        ]
        for multimodal in (False, True):
            for param, code, message in cases:
                with self.subTest(multimodal=multimodal, param=param, code=code):
                    error, requests = self.invoke(multimodal, [
                        error_response(param=param, code=code, message=message),
                        svg_response(),
                    ], openai.BadRequestError)
                    self.assertEqual(error.body["message"], message)
                    self.assertEqual(len(requests), 1)

    def test_authentication_errors_propagate_without_retry(self):
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                _, requests = self.invoke(multimodal, [
                    error_response(status=401),
                ], openai.AuthenticationError)
                self.assertEqual(len(requests), 1)

    def test_failed_fallback_propagates_second_error_without_another_retry(self):
        cases = [
            ("temperature", None, "Unsupported parameter: 'temperature' is not supported with this model."),
            ("max_output_tokens", "invalid_value", "Invalid max_output_tokens."),
        ]
        for multimodal in (False, True):
            for param, code, message in cases:
                with self.subTest(multimodal=multimodal, param=param):
                    error, requests = self.invoke(multimodal, [
                        error_response(),
                        error_response(param=param, code=code, message=message),
                    ], openai.BadRequestError)
                    self.assertEqual(error.body["message"], message)
                    self.assert_fallback_payload(requests)


if __name__ == "__main__":
    unittest.main()
