class WatermarkPositioner {
  static getPositions(width, height, length) {
    const totalPixels = width * height;
    const positions = [];
    
    // Use prime numbers for position calculation
    const primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    
    for (let i = 0; i < length; i++) {
      // Calculate position using a formula with primes
      const prime1 = primes[i % primes.length];
      const prime2 = primes[(i + 7) % primes.length];
      
      // Generate a position that depends on current index and primes
      const position = (prime1 * i + prime2) % totalPixels;
      
      positions.push(position);
    }
    
    return positions;
  }
}
module.exports = WatermarkPositioner;

