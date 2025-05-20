require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const connectDB = require('./database/connection');
const cors = require('cors');
const morgan = require('morgan');
const helmet = require('helmet');
const errorHandler = require('./app/Middleware/errorHandler');
const routes = require('./routes/api');


const app = express();


app.use(cors());
app.use(express.json());
app.use(helmet());
app.use(morgan('dev'));
app.use(express.urlencoded({ extended: true }));


connectDB();

app.use('/api', routes);

app.use('/', require('./routes/api'));

app.use((req, res, next) => {
  const error = new Error('Route not found');
  error.status = 404;
  next(error);
});


app.use(errorHandler);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));