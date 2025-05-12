function createResponse(res, statusCode, message, data = null) {
    const response = {
      success: statusCode < 400,
      message,
    };
  
    if (data !== null) {
      response.data = data;
    }
  
    return res.status(statusCode).json(response);
  }
  class AppError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.status = status;
    this.name = this.constructor.name;
  }
}
module.exports = { createResponse };
  