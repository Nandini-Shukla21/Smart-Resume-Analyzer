import { motion } from "framer-motion";
import { Heart, Brain } from "lucide-react";

export const Footer = () => {
  return (
    <motion.footer
      className="mt-20 py-8 border-t border-glass-border backdrop-blur-xl bg-glass/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="max-w-7xl mx-auto px-4 md:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="text-lg font-semibold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              AI Resume Analyzer
            </span>
          </div>

          {/* Powered By */}
          <motion.div
            className="flex items-center gap-2 text-muted-foreground"
            whileHover={{ scale: 1.05 }}
          >
            <span className="text-sm">Powered by</span>
            <Heart className="w-4 h-4 text-destructive animate-pulse" />
            <span className="text-sm font-semibold bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Nandini Shukla
            </span>
          </motion.div>

          {/* Copyright */}
          <div className="text-sm text-muted-foreground">
            © 2024 AI Resume Analyzer. All rights reserved.
          </div>
        </div>
      </div>
    </motion.footer>
  );
};