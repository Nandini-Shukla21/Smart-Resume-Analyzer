import { useState } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { GlassCard } from "@/components/ui/glass-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Header } from "@/components/ui/header";
import { ArrowLeft, Mail, Lock, User, Eye, EyeOff } from "lucide-react";
import { auth } from "@/firebase";
import { createUserWithEmailAndPassword, signInWithEmailAndPassword } from "firebase/auth";

const Auth = () => {
  const [isSignUp, setIsSignUp] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const navigate = useNavigate();

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      if (isSignUp) {
        if (formData.password !== formData.confirmPassword) {
          alert("❌ Passwords do not match!");
          return;
        }
        await createUserWithEmailAndPassword(auth, formData.email, formData.password);
        alert("✅ Account created successfully!");
      } else {
        await signInWithEmailAndPassword(auth, formData.email, formData.password);
        alert("✅ Logged in successfully!");
      }
      navigate("/dashboard");
    } catch (error: any) {
      alert("❌ " + error.message);
    }
  };

  return (
    <div
      className="min-h-screen relative overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, hsl(250 100% 98% / 0.95), hsl(280 60% 95% / 0.9))',
      }}
    >
      {/* Floating Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute -top-40 -left-40 w-80 h-80 rounded-full bg-gradient-to-br from-primary/20 to-accent/10"
          animate={{ y: [0, 30, 0], x: [0, 20, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute -bottom-40 -right-40 w-96 h-96 rounded-full bg-gradient-to-br from-secondary/15 to-accent/20"
          animate={{ y: [0, -25, 0], x: [0, -15, 0] }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 1 }}
        />
        <motion.div
          className="absolute top-1/3 right-1/4 w-32 h-32 rounded-full bg-gradient-to-br from-accent/10 to-primary/15"
          animate={{ scale: [1, 1.1, 1], rotate: [0, 180, 360] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <Header />

      <div className="min-h-screen flex items-center justify-center p-4 pt-24 relative z-10">
        <div className="max-w-md w-full">
          {/* Back Button */}
          <motion.div
            className="mb-6"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Button
              variant="ghost"
              onClick={() => navigate("/")}
              className="flex items-center gap-2 text-muted-foreground hover:text-foreground backdrop-blur-sm bg-glass/30 border border-glass-border/50"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Home
            </Button>
          </motion.div>

          {/* Auth Form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <GlassCard variant="large">
              <div className="text-center mb-8">
                <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
                  {isSignUp ? "Create Account" : "Welcome Back"}
                </h1>
                <p className="text-muted-foreground">
                  {isSignUp
                    ? "Sign up to start analyzing your resume"
                    : "Sign in to continue your journey"}
                </p>
              </div>

              <form onSubmit={handleSubmit} className="space-y-6">
                {isSignUp && (
                  <div className="space-y-2">
                    <Label htmlFor="name" className="text-sm font-medium">
                      Full Name
                    </Label>
                    <div className="relative">
                      <User className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                      <Input
                        id="name"
                        type="text"
                        placeholder="Enter your full name"
                        value={formData.name}
                        onChange={(e) => handleInputChange("name", e.target.value)}
                        className="pl-10 bg-glass/50 border-glass-border"
                        required
                      />
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="email" className="text-sm font-medium">
                    Email Address
                  </Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                    <Input
                      id="email"
                      type="email"
                      placeholder="Enter your email"
                      value={formData.email}
                      onChange={(e) => handleInputChange("email", e.target.value)}
                      className="pl-10 bg-glass/50 border-glass-border"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="password" className="text-sm font-medium">
                    Password
                  </Label>
                  <div className="relative">
                    <Lock className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                    <Input
                      id="password"
                      type={showPassword ? "text" : "password"}
                      placeholder="Enter your password"
                      value={formData.password}
                      onChange={(e) => handleInputChange("password", e.target.value)}
                      className="pl-10 pr-10 bg-glass/50 border-glass-border"
                      required
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="absolute right-2 top-2 w-6 h-6"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? (
                        <EyeOff className="w-4 h-4" />
                      ) : (
                        <Eye className="w-4 h-4" />
                      )}
                    </Button>
                  </div>
                </div>

                {isSignUp && (
                  <div className="space-y-2">
                    <Label htmlFor="confirmPassword" className="text-sm font-medium">
                      Confirm Password
                    </Label>
                    <div className="relative">
                      <Lock className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                      <Input
                        id="confirmPassword"
                        type="password"
                        placeholder="Confirm your password"
                        value={formData.confirmPassword}
                        onChange={(e) => handleInputChange("confirmPassword", e.target.value)}
                        className="pl-10 bg-glass/50 border-glass-border"
                        required
                      />
                    </div>
                  </div>
                )}

                <Button type="submit" className="w-full btn-gradient" size="lg">
                  {isSignUp ? "Create Account" : "Sign In"}
                </Button>
              </form>

              <div className="text-center mt-6">
                <p className="text-muted-foreground text-sm">
                  {isSignUp ? "Already have an account?" : "Don't have an account?"}
                  <Button
                    variant="link"
                    onClick={() => setIsSignUp(!isSignUp)}
                    className="ml-1 p-0 h-auto text-primary hover:text-primary-glow"
                  >
                    {isSignUp ? "Sign In" : "Sign Up"}
                  </Button>
                </p>
              </div>
            </GlassCard>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Auth;
