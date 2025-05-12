class PixelSelector {

  static getIndices(width, height, busyAreas = [], dataLength) {
    // If ML has identified busy areas, i prioritize them
    if (busyAreas?.length > 0) {
      return this._getOptimizedIndices(width, height, busyAreas, dataLength);
    }
    
    // else basic pixel selection with some randomization
    return this._getRandomizedIndices(width, height, dataLength);
  }

  static _getOptimizedIndices(width, height, busyAreas, dataLength) {
    // Get pixels from busy areas and sort them by "busyness" score if available
    const busyPixels = [];
    
    // Process each busy area, weighting by the area's busyness score
    for (const area of busyAreas) {
      const areaScore = area.score || 1.0; // Default score if not provided
      
      for (let y = area.y; y < area.y + area.height; y++) {
        for (let x = area.x; x < area.x + area.width; x++) {
          if (x < width && y < height) {
            busyPixels.push({
              index: y * width + x,
              score: areaScore
            });
          }
        }
      }
    }
    
    // Sort by score (higher score = better hiding spot)
    busyPixels.sort((a, b) => b.score - a.score);
    
    // Take the best spots first
    const selectedIndices = busyPixels.slice(0, dataLength).map(pixel => pixel.index);
    
    // If we need more pixels than available in busy areas
    if (selectedIndices.length < dataLength) {
      this._fillRemainingIndices(selectedIndices, width, height, busyAreas, dataLength);
    }
    
    return selectedIndices;
  }

  
  /*
    static getIndices(width, height, busyAreas = [], dataLength) {
      const indices = [];
      
      if (busyAreas?.length > 0) {
        const busyPixels = this._getBusyPixels(width, height, busyAreas);
        const shuffledBusyIndices = this._shuffleArray([...busyPixels]);
        
        indices.push(...shuffledBusyIndices.slice(0, dataLength));
        
        if (dataLength > busyPixels.size) {
          this._fillRemainingIndices(indices, width * height, busyPixels, dataLength);
        }
      } else {
        indices.push(...Array.from({length: dataLength}, (_, i) => i));
      }
      
      return indices;
    }
    static _getBusyPixels(width, height, busyAreas) {
        const busyPixels = new Set();
        
        for (const area of busyAreas) {
          for (let y = area.y; y < area.y + area.height; y++) {
            for (let x = area.x; x < area.x + area.width; x++) {
              if (x < width && y < height) {
                busyPixels.add(y * width + x);
              }
            }
          }
        }
        
        return busyPixels;
      }
    
      static _fillRemainingIndices(indices, totalPixels, busyPixels, dataLength) {
        let currentIndex = 0;
        while (indices.length < dataLength && currentIndex < totalPixels) {
          if (!busyPixels.has(currentIndex)) {
            indices.push(currentIndex);
          }
          currentIndex++;
        }
        
        if (indices.length < dataLength) {
          throw new Error('Not enough non-busy pixels to embed full message.');
        }
      }
    
      static _shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
      }
    */
}
  
  module.exports = PixelSelector;