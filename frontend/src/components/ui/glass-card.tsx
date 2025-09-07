import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  variant?: "default" | "large" | "gradient";
  hover?: boolean;
  [key: string]: any;
}

export const GlassCard = ({ 
  children, 
  className, 
  variant = "default",
  hover = true,
  ...props 
}: GlassCardProps) => {
  const variants = {
    default: "glass-card p-6",
    large: "glass-card-lg p-8",
    gradient: "glass-card p-6 bg-gradient-primary border-primary/20"
  };

  return (
    <motion.div
      className={cn(
        variants[variant],
        hover && "hover:scale-[1.02] hover:shadow-glass-lg transition-all duration-300",
        className
      )}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      {...props}
    >
      {children}
    </motion.div>
  );
};