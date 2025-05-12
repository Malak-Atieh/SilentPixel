class PixelSelector {

  static getIndices(width, height, busyAreas = [], dataLength) {
    // If ML has identified busy areas, i prioritize them
    if (busyAreas?.length > 0) {
      return this._getOptimizedIndices(width, height, busyAreas, dataLength);
    }
    
    // else basic pixel selection with some randomization
    return this._getRandomizedIndices(width, height, dataLength);
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