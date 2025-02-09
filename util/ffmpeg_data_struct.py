# -*- coding: utf-8 -*-

# ffmpeg数据结构

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class Stream(BaseModel):
    index: int = Field(..., description="流在文件中的索引")
    codec_name: str = Field(..., description="编解码器名称 (例如 h264, aac)")
    codec_type: str = Field(..., description="流类型 (例如 video, audio)")
    codec_long_name: Optional[str] = Field(None, description="编解码器全名")
    width: Optional[int] = Field(None, description="视频流的宽度（如果适用）")
    height: Optional[int] = Field(None, description="视频流的高度（如果适用）")
    sample_rate: Optional[str] = Field(None, description="音频流的采样率（如果适用）")
    channels: Optional[int] = Field(None, description="音频流的声道数（如果适用）")
    duration: Optional[str] = Field(None, description="流的时长")
    bit_rate: Optional[str] = Field(None, description="比特率")
    tags: Optional[Dict[str, str]] = Field(None, description="与流相关的元数据标签")

class Format(BaseModel):
    filename: str = Field(..., description="文件名")
    nb_streams: int = Field(..., description="文件中流的数量")
    format_name: str = Field(..., description="容器格式名称")
    format_long_name: Optional[str] = Field(None, description="容器格式全名")
    start_time: Optional[str] = Field(None, description="开始时间")
    duration: Optional[str] = Field(None, description="文件总时长")
    size: Optional[str] = Field(None, description="文件大小")
    bit_rate: Optional[str] = Field(None, description="平均比特率")
    tags: Optional[Dict[str, str]] = Field(None, description="与文件相关的元数据标签")

class FFProbeOutput(BaseModel):
    streams: List[Stream] = Field(..., description="文件中所有流的信息列表")
    format: Format = Field(..., description="文件格式信息")