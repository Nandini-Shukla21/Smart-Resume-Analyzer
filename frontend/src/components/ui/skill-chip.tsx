import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { Check, X } from "lucide-react";

interface SkillChipProps {
  skill: string;
  variant?: "matched" | "missing" | "neutral";
  className?: string;
  index?: number;
}

export const SkillChip = ({ 
  skill, 
  variant = "neutral", 
  className,
  index = 0 
}: SkillChipProps) => {
  const variants = {
    matched: "bg-success/20 text-success border-success/30",
    missing: "bg-destructive/20 text-destructive border-destructive/30",
    neutral: "bg-muted/50 text-muted-foreground border-muted"
  };

  const icons = {
    matched: <Check className="w-3 h-3" />,
    missing: <X className="w-3 h-3" />,
    neutral: null
  };

  return (
    <motion.div
      className={cn(
        "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-sm font-medium",
        "backdrop-blur-sm transition-all duration-200 hover:scale-105",
        variants[variant],
        className
      )}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ 
        duration: 0.3, 
        delay: index * 0.05,
        type: "spring",
        stiffness: 300,
        damping: 20
      }}
      whileHover={{ y: -2 }}
    >
      {icons[variant]}
      <span>{skill}</span>
    </motion.div>
  );
};