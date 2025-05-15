const crypto = require('crypto');
class PixelSelector {

  static getIndices(width, height, busyAreas = [], dataLength, password = null) {
        if (dataLength <= 0) return [];
    
    // If ML has identified busy areas, i prioritize them
    if (busyAreas?.length > 0) {
      const optimized = this._getOptimizedIndices(width, height, busyAreas, dataLength);
      if (optimized.length >= dataLength) return optimized;    
    }
    
    // else basic pixel selection with some randomization
    return password 
      ? this._getPasswordBasedIndices(width, height, dataLength, password)
      : this._getRandomizedIndices(width, height, dataLength);
  }

  static _getOptimizedIndices(width, height, busyAreas, dataLength) {
    // Get pixels from busy areas and sort them by "busyness" score if available
    const busyPixels = [];
    
    busyAreas.forEach(area => {
      const areaScore = area.score || 1.0;
      const xEnd = Math.min(area.x + area.width, width);
      const yEnd = Math.min(area.y + area.height, height);
      
      for (let y = Math.max(0, area.y); y < yEnd; y++) {
        for (let x = Math.max(0, area.x); x < xEnd; x++) {
          busyPixels.push({
            index: y * width + x,
            score: areaScore * (1 + Math.random() * 0.2) // Add slight randomness
          });
        }
      }
    });
    
    return busyPixels
      .sort((a, b) => b.score - a.score)
      .slice(0, dataLength)
      .map(p => p.index);
  }

    static _getPasswordBasedIndices(width, height, dataLength, password) {
    const totalPixels = width * height;
    const indices = [];
    const hash = crypto.createHash('sha256').update(password).digest();
    
    for (let i = 0; i < dataLength; i++) {
      const hashPart = hash.readUInt32BE(i * 4 % hash.length);
      const index = (hashPart + i) % totalPixels;
      indices.push(index);
    }
    
    return indices;
  }

  static _getRandomizedIndices(width, height, dataLength) {
    const indices = new Set();
    const totalPixels = width * height;
    
    while (indices.size < Math.min(dataLength, totalPixels)) {
      indices.add(Math.floor(Math.random() * totalPixels));
    }
    
    return Array.from(indices);
  }
  
    static _fillRemainingIndices(selectedIndices, width, height, busyAreas, dataLength) {
    // Create set of already selected indices for quick lookup
    const selectedSet = new Set(selectedIndices);
    
    // Generate array of remaining indices not in busy areas
    const remainingIndices = [];
    for (let i = 0; i < width * height; i++) {
      if (!selectedSet.has(i) && !this._isInBusyAreas(i, width, busyAreas)) {
        remainingIndices.push(i);
      }
    }
    
    // Shuffle remaining indices for randomness
    const shuffled = this._shuffleArray(remainingIndices);
    
    // Add remaining indices until we reach required data length
    for (let i = 0; i < shuffled.length && selectedIndices.length < dataLength; i++) {
      selectedIndices.push(shuffled[i]);
    }
    
    if (selectedIndices.length < dataLength) {
      throw new Error('Not enough pixels to embed the full message');
    }
  }

  static _isInBusyAreas(index, width, busyAreas) {
    const x = index % width;
    const y = Math.floor(index / width);
    
    for (const area of busyAreas) {
      if (x >= area.x && x < area.x + area.width && 
          y >= area.y && y < area.y + area.height) {
        return true;
      }
    }
    return false;
  }

  static _shuffleArray(array) {
    const result = [...array];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}

module.exports = PixelSelector;