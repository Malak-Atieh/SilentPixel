class PixelSelector {
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
  
}
  
  module.exports = PixelSelector;