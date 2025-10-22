// AI Resume Analyzer API Service
// Handles all backend communication for resume analysis

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000";

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  // Generic GET (for JSON responses)
  
  async getRequest(endpoint) {
    const url = `${this.baseURL}${endpoint}`;
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error("API GET failed:", error);
      throw error;
    }
  }

  // Generic POST (for JSON body)
  
  async postRequest(endpoint, body) {
    const url = `${this.baseURL}${endpoint}`;
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error("API POST failed:", error);
      throw error;
    }
  }

  // ✅ Upload and analyze resume
  
 async analyzeResume(file, jobRole = null) {
  const formData = new FormData();
  formData.append("resume", file);
  if (jobRole) formData.append("job_role", jobRole);

  try {
    const response = await fetch(`${this.baseURL}/analyze`, {
      method: "POST",
      body: formData,
    });

    // DEBUG: log raw text
    
    const text = await response.text();
    console.log("Backend /analyze response:", text);

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.status} - ${text}`);
    }

    return JSON.parse(text);
  } catch (error) {
    console.error("Analyze failed:", error);
    throw error;
  }
}


  // ✅ Get resume scoring breakdown

  async getResumeScore(resumeId) {
    return this.getRequest(`/score/${resumeId}`);
  }

  // ✅ Get keyword analysis
  
  async getKeywordAnalysis(resumeId, jobRole = null) {
    const endpoint = jobRole
      ? `/keywords/${resumeId}?job_role=${encodeURIComponent(jobRole)}` // FIXED
      : `/keywords/${resumeId}`;
    return this.getRequest(endpoint);
  }

  // ✅ Get grammar and readability analysis
  
  async getGrammarAnalysis(resumeId) {
    return this.getRequest(`/grammar/${resumeId}`);
  }

  // ✅ Compare resume with job role
  
  async compareWithJob(resumeId, jobRole) {
    return this.postRequest(`/compare/${resumeId}`, { job_role: jobRole }); // FIXED
  }

  // ✅ Export analysis report (frontend-only for now)
  
  async exportReport(analysisData) {
    return Promise.resolve(analysisData);
  }

  // ✅ Mock data for UI testing
  
  getMockAnalysisData() {
    return {
      overall_score: 78,
      ats_score: 82,
      keyword_match_percentage: 75,
      grammar_score: 85,
      readability_score: 80,
      matched_skills: ["JavaScript", "React", "Node.js", "Python", "AWS", "Docker"],
      missing_skills: ["Kubernetes", "TypeScript", "GraphQL", "CI/CD"],
      grammar_suggestions: [],
      keyword_analysis: {
        total_keywords: 45,
        matched_keywords: 34,
        keyword_density: 0.12,
      },
      sections_analysis: {
        experience: { score: 85 },
        skills: { score: 70 },
        education: { score: 90 },
        summary: { score: 75 },
      },
    };
  }
}

export default new ApiService();
