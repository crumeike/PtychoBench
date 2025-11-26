#!/usr/bin/env python3
"""
Data loader for PtychoBench training.
Handles image loading and data split preparation for VLM and LLM training.
"""

import json
import os
from PIL import Image
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class PtychographyDataLoader:
    """Data loader that handles image loading and text preprocessing"""
    
    def __init__(self, image_base_path: str):
        """
        Initialize data loader
        
        Args:
            image_base_path: Base directory containing ptychography images
        """
        self.image_base_path = Path(image_base_path)
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image from path
        
        Args:
            image_path: Relative or absolute path to image
            
        Returns:
            PIL Image in RGB mode, or None if loading fails
        """
        # Handle different path formats
        if image_path.startswith('/data/upload/'):
            # Strip any prefix paths
            image_name = Path(image_path).name
            full_path = self.image_base_path / image_name
        else:
            full_path = self.image_base_path / Path(image_path).name
        
        try:
            image = Image.open(full_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None
    
    def load_split_with_images(self, split_path: str) -> List[Dict]:
        """
        Load data split with images for VLM training
        
        Args:
            split_path: Path to JSON file (train.json, val.json, or test.json)
            
        Returns:
            List of samples with PIL Images loaded
        """
        with open(split_path, 'r') as f:
            data = json.load(f)
        
        loaded_data = []
        
        for sample in data:
            messages = sample["messages"]
            user_content = messages[0]["content"]
            
            # Convert image paths to PIL Images
            new_content = []
            for item in user_content:
                if item["type"] == "image":
                    if "image_path" in item:
                        image = self.load_image(item["image_path"])
                        if image is not None:
                            new_content.append({
                                "type": "image",
                                "image": image
                            })
                        else:
                            print(f"Warning: Could not load image {item['image_path']}")
                            continue
                    else:
                        # Already has image object
                        new_content.append(item)
                else:
                    new_content.append(item)
            
            # Only include sample if image was successfully loaded
            if any(item["type"] == "image" for item in new_content):
                loaded_sample = {
                    "messages": [
                        {"role": "user", "content": new_content},
                        {"role": "assistant", "content": messages[1]["content"]}
                    ]
                }
                
                # Preserve metadata if present
                if "metadata" in sample:
                    loaded_sample["metadata"] = sample["metadata"]
                    
                loaded_data.append(loaded_sample)
        
        return loaded_data
    
    def load_split_text_only(self, split_path: str) -> List[Dict]:
        """
        Load data split as text-only for LLM training
        
        Args:
            split_path: Path to JSON file
            
        Returns:
            List of text-only samples (no images)
        """
        with open(split_path, 'r') as f:
            data = json.load(f)
        
        loaded_data = []
        
        for sample in data:
            messages = sample["messages"]
            user_content = messages[0]["content"]
            
            # Extract only text content, ignore images
            text_parts = []
            for item in user_content:
                if item["type"] == "text":
                    text_parts.append(item["text"])
            
            combined_text = " ".join(text_parts).strip()
            
            # Only include if there's text content
            if combined_text:
                loaded_sample = {
                    "messages": [
                        {"role": "user", "content": combined_text},
                        {"role": "assistant", "content": messages[1]["content"][0]["text"]}
                    ]
                }
                
                # Preserve metadata if present
                if "metadata" in sample:
                    loaded_sample["metadata"] = sample["metadata"]
                    
                loaded_data.append(loaded_sample)
        
        return loaded_data
    
    def load_all_splits(self, data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load all splits (train, val, test) with images for VLM training
        
        Args:
            data_dir: Directory containing train.json, val.json, test.json
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_path = os.path.join(data_dir, f"{split_name}.json")
            if os.path.exists(split_path):
                splits[split_name] = self.load_split_with_images(split_path)
                print(f"Loaded {split_name}: {len(splits[split_name])} samples")
            else:
                print(f"Warning: {split_path} not found")
                splits[split_name] = []
        
        return splits['train'], splits['val'], splits['test']
    
    def load_all_splits_text_only(self, data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load all splits as text-only for LLM training
        
        Args:
            data_dir: Directory containing train.json, val.json, test.json
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_path = os.path.join(data_dir, f"{split_name}.json")
            if os.path.exists(split_path):
                splits[split_name] = self.load_split_text_only(split_path)
                print(f"Loaded {split_name} (text-only): {len(splits[split_name])} samples")
            else:
                print(f"Warning: {split_path} not found")
                splits[split_name] = []
        
        return splits['train'], splits['val'], splits['test']
