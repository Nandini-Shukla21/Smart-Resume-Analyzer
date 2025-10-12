import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// Replace with your Firebase config (from Firebase Console)

const firebaseConfig = {
  apiKey: "AIzaSyCW3MqG4nZA7ZYk4BXKxzFjnYfnKlqyU30",
  authDomain: "ai-resume-analyzer-e01e1.firebaseapp.com",
  projectId: "ai-resume-analyzer-e01e1",
  storageBucket: "ai-resume-analyzer-e01e1.appspot.com",
  messagingSenderId: "850290121597",
  appId: "1:850290121597:web:e2aed469c96088b77d4a14",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
