import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { FileUpload } from "@/components/ui/file-upload";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Header } from "@/components/ui/header";
import { Footer } from "@/components/ui/footer";
import { Brain, Zap, Target, FileSearch } from "lucide-react";
import apiService from "@/services/api";

// ✅ import job_roles list
import jobKeywords from "../../../job_keywords.json";

const Home = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [jobRole, setJobRole] = useState("");
  const navigate = useNavigate();

  const handleFileUpload = async (file: File) => {
    if (!jobRole) {
      alert("Please select a job role before uploading your resume.");
      return;
    }

    setIsAnalyzing(true);

    try {
      // ✅ Send only role name
      const analyzeRes = await apiService.analyzeResume(file, jobRole);
      const resumeId = analyzeRes.resume_id;

      // ✅ Fetch extra details in parallel
      const [scoreData, keywordData, grammarData] = await Promise.all([
        apiService.getResumeScore(resumeId),
        apiService.getKeywordAnalysis(resumeId, jobRole), // pass role name
        apiService.getGrammarAnalysis(resumeId),
      ]);

      // ✅ Shape backend response into Dashboard format
      const analysisData = {
        resume_id: resumeId,
        prediction: analyzeRes.prediction,
        job_role: jobRole,

        // Scores
        overall_score: scoreData.overall_score,
        ats_score: scoreData.ats_score,
        grammar_score: grammarData.score,
        keyword_match_percentage:
          keywordData.total_keywords > 0
            ? Math.round(
                (keywordData.matched_keywords / keywordData.total_keywords) * 100
              )
            : 0,

        // Skills & Keywords
        matched_skills: keywordData.matched,
        missing_skills: keywordData.missing,
        keyword_analysis: {
          total_keywords: keywordData.total_keywords,
          matched_keywords: keywordData.matched_keywords,
          keyword_density: keywordData.keyword_density,
        },

        // Grammar
        grammar_suggestions: grammarData.suggestions.map((s: any) => ({
          text: s.text,
          suggestion: s.suggestion || "",
        })),

        // Sections
        sections_analysis: scoreData.sections_analysis || {
          experience: { score: 80, suggestions: [] },
          skills: { score: 75, suggestions: [] },
          education: { score: 85, suggestions: [] },
          summary: { score: 70, suggestions: [] },
        },

        // Readability dummy
        readability_score: 80,
      };

      // ✅ Navigate to dashboard with shaped data
      navigate("/dashboard", { state: { analysisData } });
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Failed to analyze resume. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const features = [
    {
      icon: <Brain className="w-6 h-6" />,
      title: "AI-Powered Analysis",
      description:
        "Advanced algorithms analyze your resume for ATS compatibility and optimization opportunities.",
    },
    {
      icon: <Target className="w-6 h-6" />,
      title: "Keyword Matching",
      description:
        "Compare your skills with job requirements and identify missing keywords.",
    },
    {
      icon: <FileSearch className="w-6 h-6" />,
      title: "Grammar & Readability",
      description:
        "Comprehensive grammar check and readability analysis for professional polish.",
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "Instant Results",
      description:
        "Get detailed feedback and actionable insights in seconds, not hours.",
    },
  ];

  const jobRoles = Object.keys(jobKeywords);

  return (
    <div className="min-h-screen relative">
      <div className="home-background" />
      <Header />

      <div className="px-4 md:px-8 pt-8">
        <div className="max-w-6xl mx-auto">
          {/* Hero Section */}
          <motion.div
            className="text-center mb-16"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <div className="float-animation">
              <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
                AI Resume Analyzer
              </h1>
            </div>

            <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              Transform your resume with AI-powered analysis. Get instant
              feedback on ATS compatibility, keyword optimization, and
              professional presentation.
            </p>
          </motion.div>

          {/* Upload Section */}
          <motion.div
            className="mb-16"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <GlassCard variant="large" className="max-w-2xl mx-auto">
              <div className="text-center mb-6">
                <h2 className="text-2xl font-semibold mb-2">Upload Your Resume</h2>
                <p className="text-muted-foreground">
                  Select a job role and upload your resume in PDF or DOCX format
                </p>
              </div>

              {/* Job Role Selector */}
              <div className="mb-6 relative">
                <select
                  value={jobRole}
                  onChange={(e) => setJobRole(e.target.value)}
                  className="w-full p-3 rounded-lg border border-white/20
                             bg-white/90 text-black 
                             dark:bg-white/90 dark:text-black
                             focus:outline-none focus:ring-2 focus:ring-[hsl(300,100%,60%)]
                             appearance-none transition"
                >
                  <option value="">-- Select Job Role --</option>
                  {jobRoles.map((role) => (
                    <option
                      key={role}
                      value={role}
                      className="text-black hover:bg-[hsl(300,100%,60%)] hover:text-white cursor-pointer"
                    >
                      {role}
                    </option>
                  ))}
                </select>

                {/* custom dropdown arrow */}
                <span className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-black">
                  ▼
                </span>
              </div>

              {/* Resume Upload */}
              <FileUpload
                onFileSelect={handleFileUpload}
                placeholder={
                  isAnalyzing ? "Analyzing your resume..." : "Upload your resume"
                }
              />

              {isAnalyzing && (
                <motion.div
                  className="mt-6 text-center"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <div className="pulse-glow glass-card p-4 inline-block">
                    <p className="text-primary font-medium">
                      🧠 AI is analyzing your resume...
                    </p>
                  </div>
                </motion.div>
              )}
            </GlassCard>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
              >
                <GlassCard className="h-full text-center feature-card cursor-pointer">
                  <div className="text-primary mb-4 flex justify-center">
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-muted-foreground text-sm">
                    {feature.description}
                  </p>
                </GlassCard>
              </motion.div>
            ))}
          </motion.div>

          {/* CTA Section */}
          <motion.div
            className="text-center mt-16"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <GlassCard variant="gradient" className="max-w-2xl mx-auto">
              <h3 className="text-2xl font-bold mb-4 text-white">
                Ready to optimize your resume?
              </h3>
              <p className="text-white/90 mb-6">
                Join thousands of professionals who have improved their job
                prospects with our AI-powered analysis.
              </p>
              <Button
                className="btn-glass text-white border-white/30 hover:bg-white/20"
                size="lg"
                onClick={() => navigate("/auth")}
              >
                Get Started Free
              </Button>
            </GlassCard>
          </motion.div>
        </div>
      </div>

      <Footer />
      
    </div>
  );
};

export default Home;
