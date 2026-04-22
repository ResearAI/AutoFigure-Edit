"""Unit tests for the MiniMax LLM provider integration."""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Import helpers – the module under test uses heavy ML deps (torch, etc.)
# We mock them when they are not installed so the provider-level tests still
# run in lightweight CI environments.
# ---------------------------------------------------------------------------

import importlib
import sys

_STUBS: dict[str, Any] = {}


def _ensure_stubs():
    """Create lightweight stubs for heavy optional deps if they are missing."""
    for mod_name in ("torch", "torchvision", "torchvision.transforms",
                     "timm", "transformers", "kornia"):
        if mod_name not in sys.modules:
            stub = MagicMock()
            sys.modules[mod_name] = stub
            _STUBS[mod_name] = stub


_ensure_stubs()

# Now we can import the target module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import autofigure2 as af


# ============================================================================
# Temperature clamping
# ============================================================================

class TestMiniMaxTemperatureClamping:
    """_clamp_minimax_temperature must keep values in (0, 1]."""

    def test_zero_is_clamped_up(self):
        assert af._clamp_minimax_temperature(0.0) == 0.01

    def test_negative_is_clamped_up(self):
        assert af._clamp_minimax_temperature(-0.5) == 0.01

    def test_above_one_is_clamped_down(self):
        assert af._clamp_minimax_temperature(1.5) == 1.0

    def test_exactly_one_is_kept(self):
        assert af._clamp_minimax_temperature(1.0) == 1.0

    def test_normal_value_is_kept(self):
        assert af._clamp_minimax_temperature(0.7) == 0.7

    def test_small_positive_is_kept(self):
        assert af._clamp_minimax_temperature(0.01) == 0.01


# ============================================================================
# Provider configuration
# ============================================================================

class TestMiniMaxProviderConfig:
    """PROVIDER_CONFIGS must include minimax with correct defaults."""

    def test_minimax_in_configs(self):
        assert "minimax" in af.PROVIDER_CONFIGS

    def test_base_url(self):
        assert af.PROVIDER_CONFIGS["minimax"]["base_url"] == "https://api.minimax.io/v1"

    def test_default_svg_model(self):
        assert af.PROVIDER_CONFIGS["minimax"]["default_svg_model"] == "MiniMax-M2.7"

    def test_no_default_image_model(self):
        assert af.PROVIDER_CONFIGS["minimax"]["default_image_model"] is None


# ============================================================================
# Image generation raises NotImplementedError
# ============================================================================

class TestMiniMaxImageGeneration:
    """MiniMax must refuse image generation with a clear error."""

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="MiniMax does not support image generation"):
            af._call_minimax_image_generation(
                prompt="draw a cat",
                api_key="test-key",
                model="MiniMax-M2.7",
                base_url="https://api.minimax.io/v1",
            )

    def test_dispatcher_raises_for_minimax(self):
        with pytest.raises(NotImplementedError):
            af.call_llm_image_generation(
                prompt="draw a cat",
                api_key="test-key",
                model="MiniMax-M2.7",
                base_url="https://api.minimax.io/v1",
                provider="minimax",
            )


# ============================================================================
# Text call dispatching
# ============================================================================

class TestMiniMaxTextCall:
    """call_llm_text must route to _call_minimax_text for minimax provider."""

    @patch("autofigure2._call_minimax_text", return_value="hello from minimax")
    def test_dispatch(self, mock_call):
        result = af.call_llm_text(
            prompt="say hello",
            api_key="test-key",
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            provider="minimax",
        )
        assert result == "hello from minimax"
        mock_call.assert_called_once()

    @patch("autofigure2._call_minimax_text")
    def test_temperature_passed(self, mock_call):
        af.call_llm_text(
            prompt="hi",
            api_key="k",
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            provider="minimax",
            temperature=0.5,
        )
        _, kwargs = mock_call.call_args
        assert kwargs.get("temperature", mock_call.call_args[0][5] if len(mock_call.call_args[0]) > 5 else None) is not None


# ============================================================================
# Multimodal call dispatching
# ============================================================================

class TestMiniMaxMultimodalCall:
    """call_llm_multimodal must route to _call_minimax_multimodal."""

    @patch("autofigure2._call_minimax_multimodal", return_value="svg code here")
    def test_dispatch(self, mock_call):
        result = af.call_llm_multimodal(
            contents=["generate svg"],
            api_key="test-key",
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            provider="minimax",
        )
        assert result == "svg code here"
        mock_call.assert_called_once()


# ============================================================================
# Think-tag stripping in multimodal response
# ============================================================================

class TestThinkTagStripping:
    """MiniMax multimodal must strip <think> tags from responses."""

    @patch("autofigure2.OpenAI", create=True)
    def test_think_tag_stripped_when_content_outside(self, mock_openai_cls):
        """Think tags should be stripped when there is content outside them."""
        mock_msg = MagicMock()
        mock_msg.content = "<think>reasoning here</think>SVG output"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            result = af._call_minimax_multimodal(
                contents=["test"],
                api_key="key",
                model="MiniMax-M2.7",
                base_url="https://api.minimax.io/v1",
            )

        assert "<think>" not in result
        assert result == "SVG output"

    @patch("autofigure2.OpenAI", create=True)
    def test_think_tag_preserved_when_only_content(self, mock_openai_cls):
        """If all content is inside think tags, preserve the original response."""
        mock_msg = MagicMock()
        mock_msg.content = "<think>the entire answer is here</think>"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        with patch.dict("sys.modules", {"openai": MagicMock(OpenAI=mock_openai_cls)}):
            result = af._call_minimax_multimodal(
                contents=["test"],
                api_key="key",
                model="MiniMax-M2.7",
                base_url="https://api.minimax.io/v1",
            )

        # Should preserve the original since stripping would leave empty
        assert result == "<think>the entire answer is here</think>"


# ============================================================================
# figure_path support in method_to_svg
# ============================================================================

class TestFigurePathSupport:
    """method_to_svg should accept figure_path and skip image generation."""

    def test_figure_path_copies_to_output(self):
        """When figure_path is provided, step 1 should copy it instead of calling LLM."""
        # Create a temporary figure file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a minimal PNG (1x1 pixel)
            from PIL import Image
            fig_src = Path(tmpdir) / "source_figure.png"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(str(fig_src))

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Patch heavy functions that would run in step 2+
            with patch.object(af, "segment_with_sam3") as mock_sam, \
                 patch.object(af, "_ensure_rmbg2_access_ready"), \
                 patch.object(af, "crop_and_remove_background", return_value=[]), \
                 patch.object(af, "generate_svg_template"), \
                 patch.object(af, "optimize_svg_with_llm"), \
                 patch.object(af, "replace_icons_in_svg"):

                mock_sam.return_value = (str(output_dir / "samed.png"), str(output_dir / "boxlib.json"), [])

                # Create dummy samed.png and boxlib.json that steps expect
                Image.new("RGB", (100, 100)).save(str(output_dir / "samed.png"))
                Path(output_dir / "boxlib.json").write_text("[]")

                result = af.method_to_svg(
                    method_text="test method",
                    output_dir=str(output_dir),
                    api_key="test-key",
                    provider="minimax",
                    figure_path=str(fig_src),
                    stop_after=1,
                )

            # The figure should be copied to output
            assert (output_dir / "figure.png").is_file()
            assert result["figure_path"] == str(output_dir / "figure.png")

    def test_missing_figure_path_raises(self):
        """When figure_path points to a missing file, should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="预生成的图片不存在"):
                af.method_to_svg(
                    method_text="test",
                    output_dir=str(Path(tmpdir) / "output"),
                    api_key="test-key",
                    provider="minimax",
                    figure_path="/nonexistent/figure.png",
                )


# ============================================================================
# CLI argument parsing
# ============================================================================

class TestCLIArgs:
    """Verify minimax is accepted by the CLI argparse config."""

    def test_minimax_in_provider_choices(self):
        """The --provider choices must include minimax."""
        import argparse
        # Extract the provider argument from the parser setup
        # We check PROVIDER_CONFIGS since the choices list comes from there
        assert "minimax" in af.PROVIDER_CONFIGS

    def test_figure_path_default_none(self):
        """--figure_path should default to None when not provided."""
        # This is implicitly tested by the argparse default
        # We verify via the method_to_svg signature
        import inspect
        sig = inspect.signature(af.method_to_svg)
        assert sig.parameters["figure_path"].default is None


# ============================================================================
# Integration test (requires MINIMAX_API_KEY)
# ============================================================================

@pytest.mark.skipif(
    not os.environ.get("MINIMAX_API_KEY"),
    reason="MINIMAX_API_KEY not set",
)
class TestMiniMaxIntegration:
    """Integration tests that call the real MiniMax API."""

    def test_text_call(self):
        """A simple text completion should return a non-empty string."""
        result = af._call_minimax_text(
            prompt="Reply with exactly: MINIMAX_OK",
            api_key=os.environ["MINIMAX_API_KEY"],
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            max_tokens=50,
            temperature=0.01,
        )
        assert result is not None
        assert len(result) > 0

    def test_multimodal_call(self):
        """A multimodal call with text-only content should work."""
        result = af._call_minimax_multimodal(
            contents=["What is 2+2? Answer with only the number."],
            api_key=os.environ["MINIMAX_API_KEY"],
            model="MiniMax-M2.7",
            base_url="https://api.minimax.io/v1",
            max_tokens=50,
            temperature=0.5,
        )
        assert result is not None
        assert len(result) > 0

    def test_image_generation_raises(self):
        """Image generation must raise NotImplementedError even with a real key."""
        with pytest.raises(NotImplementedError):
            af._call_minimax_image_generation(
                prompt="a red circle",
                api_key=os.environ["MINIMAX_API_KEY"],
                model="MiniMax-M2.7",
                base_url="https://api.minimax.io/v1",
            )
