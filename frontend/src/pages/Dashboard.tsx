import { useLocation, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { GlassCard } from "@/components/ui/glass-card";
import { ProgressRing } from "@/components/ui/progress-ring";
import { SkillChip } from "@/components/ui/skill-chip";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Header } from "@/components/ui/header";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer, PieChart, Pie, Cell, Tooltip } from "recharts";
import { Download, FileText, AlertCircle, CheckCircle, TrendingUp, ArrowLeft } from "lucide-react";
import { useState } from "react";

const Dashboard = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const analysisData = location.state?.analysisData;
  const [activeTab, setActiveTab] = useState("overview");

  if (!analysisData) {
    return (
      <div className="min-h-screen">
        <Header />
        <div className="min-h-screen flex items-center justify-center p-4 pt-24">
          <GlassCard className="text-center max-w-md">
            <AlertCircle className="w-12 h-12 text-warning mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-2">No Analysis Data</h2>
            <p className="text-muted-foreground mb-4">
              Please upload a resume first to see the analysis results.
            </p>
            <Button onClick={() => navigate('/')} className="btn-gradient">
              Upload Resume
            </Button>
          </GlassCard>
        </div>
      </div>
    );
  }

  const sectionScores = [
    { name: 'Experience', score: analysisData.sections_analysis.experience.score },
    { name: 'Skills', score: analysisData.sections_analysis.skills.score },
    { name: 'Education', score: analysisData.sections_analysis.education.score },
    { name: 'Summary', score: analysisData.sections_analysis.summary.score },
  ];

  const pieData = [
    { name: 'Matched Keywords', value: analysisData.keyword_analysis.matched_keywords, color: '#10b981' },
    { name: 'Missing Keywords', value: analysisData.keyword_analysis.total_keywords - analysisData.keyword_analysis.matched_keywords, color: '#ef4444' }
  ];

  const exportReport = () => {
    // For demo purposes - would implement actual PDF export
    console.log('Exporting report...', analysisData);
  };

  return (
    <div 
      className="min-h-screen"
      style={{
        background: 'var(--gradient-dashboard)'
      }}
    >
      <Header />
      
      <div className="p-4 md:p-8 pt-24">
        <div className="max-w-7xl mx-auto">
          {/* Back Button & Header */}
          <motion.div
            className="mb-8"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Button
              onClick={() => navigate('/')}
              variant="ghost"
              className="flex items-center gap-2 mb-6 border"
              style={{
                backgroundColor: 'var(--dashboard-card)',
                color: 'hsl(var(--dashboard-primary))',
                borderColor: 'hsl(var(--dashboard-primary) / 0.2)'
              }}
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Button>
            
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
              <div>
                <h1 className="text-3xl md:text-4xl font-bold mb-2">Resume Analysis</h1>
                <p className="text-muted-foreground">Comprehensive AI-powered evaluation of your resume</p>
              </div>
              
              <Button 
                onClick={exportReport}
                className="btn-gradient flex items-center gap-2"
              >
                <Download className="w-4 h-4" />
                Export Report
              </Button>
            </div>
          </motion.div>

        {/* Score Overview */}
        <motion.div
          className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <GlassCard className="text-center hover:scale-105 transition-transform duration-300">
            <ProgressRing 
              progress={analysisData.overall_score} 
              label="Overall Score"
              color="primary"
              size={100}
            />
          </GlassCard>
          
          <GlassCard className="text-center hover:scale-105 transition-transform duration-300">
            <ProgressRing 
              progress={analysisData.ats_score} 
              label="ATS Score"
              color="success"
              size={100}
            />
          </GlassCard>
          
          <GlassCard className="text-center hover:scale-105 transition-transform duration-300">
            <ProgressRing 
              progress={analysisData.keyword_match_percentage} 
              label="Keyword Match"
              color="accent"
              size={100}
            />
          </GlassCard>
          
          <GlassCard className="text-center hover:scale-105 transition-transform duration-300">
            <ProgressRing 
              progress={analysisData.grammar_score} 
              label="Grammar Score"
              color="warning"
              size={100}
            />
          </GlassCard>
        </motion.div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="lg:col-span-2 space-y-6">
            {/* Section Scores Chart */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <GlassCard className="hover:shadow-glass-lg transition-all duration-300">
                <div className="flex items-center gap-2 mb-6">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  <h3 className="text-xl font-semibold">Section Performance</h3>
                </div>
                
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={sectionScores}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis 
                        dataKey="name" 
                        tick={{ fill: 'hsl(var(--muted-foreground))' }}
                        axisLine={{ stroke: 'hsl(var(--border))' }}
                      />
                      <YAxis 
                        tick={{ fill: 'hsl(var(--muted-foreground))' }}
                        axisLine={{ stroke: 'hsl(var(--border))' }}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'hsl(var(--glass))',
                          border: '1px solid hsl(var(--glass-border))',
                          borderRadius: '12px',
                          backdropFilter: 'blur(16px)'
                        }}
                      />
                      <Bar 
                        dataKey="score" 
                        fill="hsl(var(--primary))" 
                        radius={[4, 4, 0, 0]}
                        className="hover:opacity-80 transition-opacity cursor-pointer"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </GlassCard>
            </motion.div>

            {/* Skills Analysis */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <GlassCard>
                <h3 className="text-xl font-semibold mb-6">Skills & Keywords Analysis</h3>
                
                <div className="space-y-6">
                  <div>
                    <h4 className="text-lg font-medium mb-3 text-success">Matched Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {analysisData.matched_skills.map((skill: string, index: number) => (
                        <SkillChip 
                          key={skill} 
                          skill={skill} 
                          variant="matched" 
                          index={index}
                        />
                      ))}
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-lg font-medium mb-3 text-destructive">Missing Skills</h4>
                    <div className="flex flex-wrap gap-2">
                      {analysisData.missing_skills.map((skill: string, index: number) => (
                        <SkillChip 
                          key={skill} 
                          skill={skill} 
                          variant="missing" 
                          index={index}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Keyword Distribution */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <GlassCard className="hover:shadow-glass-lg transition-all duration-300">
                <h3 className="text-lg font-semibold mb-4">Keyword Distribution</h3>
                
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        dataKey="value"
                        className="hover:scale-105 transition-transform cursor-pointer"
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: 'hsl(var(--glass))',
                          border: '1px solid hsl(var(--glass-border))',
                          borderRadius: '12px',
                          backdropFilter: 'blur(16px)'
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="space-y-2 mt-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-success rounded-full"></div>
                      <span className="text-sm">Matched</span>
                    </div>
                    <span className="text-sm font-medium">{analysisData.keyword_analysis.matched_keywords}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-destructive rounded-full"></div>
                      <span className="text-sm">Missing</span>
                    </div>
                    <span className="text-sm font-medium">
                      {analysisData.keyword_analysis.total_keywords - analysisData.keyword_analysis.matched_keywords}
                    </span>
                  </div>
                </div>
              </GlassCard>
            </motion.div>

            {/* Grammar Suggestions */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.5 }}
            >
              <GlassCard>
                <h3 className="text-lg font-semibold mb-4">Grammar & Style</h3>
                
                <div className="space-y-3">
                  {analysisData.grammar_suggestions.map((suggestion: any, index: number) => (
                    <motion.div
                      key={index}
                      className="p-3 bg-warning/10 border border-warning/20 rounded-lg"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.1 * index }}
                    >
                      <div className="flex items-start gap-2">
                        <AlertCircle className="w-4 h-4 text-warning mt-0.5 flex-shrink-0" />
                        <div className="text-sm">
                          <p className="font-medium">{suggestion.text}</p>
                          <p className="text-muted-foreground mt-1">{suggestion.suggestion}</p>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
                
                {analysisData.grammar_suggestions.length === 0 && (
                  <div className="flex items-center gap-2 text-success">
                    <CheckCircle className="w-4 h-4" />
                    <span className="text-sm">No grammar issues found!</span>
                  </div>
                )}
              </GlassCard>
            </motion.div>

            {/* Quick Stats */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <GlassCard>
                <h3 className="text-lg font-semibold mb-4">Quick Stats</h3>
                
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Readability Score</span>
                      <span>{analysisData.readability_score}%</span>
                    </div>
                    <Progress value={analysisData.readability_score} className="h-2" />
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>Keyword Density</span>
                      <span>{(analysisData.keyword_analysis.keyword_density * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={analysisData.keyword_analysis.keyword_density * 100} className="h-2" />
                  </div>
                  
                  <div className="pt-2 border-t border-glass-border">
                    <div className="text-sm text-muted-foreground">
                      Total Keywords: {analysisData.keyword_analysis.total_keywords}
                    </div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;