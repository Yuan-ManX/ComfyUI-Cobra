from .nodes import LoadCobraModel, ExtractLineArt, GetColorValue, DrawColorHint, ColorizeImage

NODE_CLASS_MAPPINGS = {
    "LoadCobraModel": LoadCobraModel,
    "ExtractLineArt": ExtractLineArt,
    "GetColorValue": GetColorValue,
    "DrawColorHint": DrawColorHint,
    "ColorizeImage": ColorizeImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadCobraModel": "Load Cobra Model",
    "ExtractLineArt": "Extract Line Art",
    "GetColorValue": "Get Color Value",
    "DrawColorHint": "Draw Color Hint",
    "ColorizeImage": "Colorize Image",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
